from dataclasses import dataclass
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast_helper import (
    get_historical_weather,
    get_production_data,
    get_weather_forecast,
)


TIMEZONE = "Europe/Istanbul"
START_DATE = "2022-01-01"

PLANT_COORDINATES = [
    [37.76967344769773, 33.574324027291595],
    [37.797667446516925, 34.46742393001458],
]

WEATHER_FORECAST_MODELS = ["ecmwf_ifs025"]
WEATHER_VARIABLES = [
    "temperature_2m",
    "shortwave_radiation",
    "cloudcover",
    "relativehumidity_2m",
    "weathercode",
]

LOG_FEATURES = [
    "log_max_radiation",
    "log_sun_rt_lag_3days_sum",
    "log_temperature_2m_mean",
    "log_shortwave_radiation_mean",
    "log_cloudcover_mean",
    "log_relativehumidity_2m_mean",
    "log_weathercode_mean",
    "log_effective_radiation",
]


@dataclass(frozen=True)
class ForecastConfig:
    forecast_date: str
    start_date: str = START_DATE
    timezone: str = TIMEZONE

    @property
    def forecast_end(self) -> pd.Timestamp:
        return pd.to_datetime(self.forecast_date) + pd.DateOffset(days=1, hours=-1)


def default_forecast_date() -> str:
    """Forecast tomorrow by default."""
    return str(datetime.now() + pd.DateOffset(days=1))[:10]


def normalize_dt(series: pd.Series, timezone: str) -> pd.Series:
    dt = pd.to_datetime(series)

    if dt.dt.tz is None:
        return dt.dt.tz_localize(timezone)

    return dt.dt.tz_convert(timezone)


def fetch_weather_data() -> pd.DataFrame:
    historical = get_historical_weather(
        start_date=START_DATE,
        variables=WEATHER_VARIABLES,
        coordinates=PLANT_COORDINATES,
        get_forecast_data=True,
    ).dropna()

    future = get_weather_forecast(
        forecast_days=6,
        past_days=30,
        variables=WEATHER_VARIABLES,
        coordinates=PLANT_COORDINATES,
        models=WEATHER_FORECAST_MODELS,
    ).dropna()

    historical.insert(0, "source", "historical")
    future.insert(0, "source", "future")

    weather = pd.concat([historical, future], ignore_index=True)
    weather["priority"] = weather["source"].map({"historical": 0, "future": 1})

    return (
        weather.sort_values(["dt", "priority"])
        .drop_duplicates("dt", keep="first")
        .drop(columns=["source", "priority"])
        .sort_values("dt")
        .reset_index(drop=True)
    )


def build_hourly_dataset(config: ForecastConfig) -> pd.DataFrame:
    production = get_production_data()
    production["dt"] = normalize_dt(production["dt"], config.timezone)

    weather = fetch_weather_data()
    weather["dt"] = normalize_dt(weather["dt"], config.timezone)

    hourly_dates = pd.date_range(
        config.start_date,
        config.forecast_end,
        freq="1h",
        tz=config.timezone,
    )
    df = pd.DataFrame({"dt": hourly_dates})

    df = df.merge(production, on="dt", how="left")
    df = df.merge(weather, on="dt", how="left")

    df["sun_rt_lag_3days"] = df["sun_rt"].shift(3 * 24)
    df["Date"] = df["dt"].dt.tz_localize(None).dt.normalize()
    df["Hour"] = df["dt"].dt.hour

    return add_hourly_weather_features(df)


def add_hourly_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temperature_2m_mean"] = df[
        ["location_000 temperature_2m", "location_001 temperature_2m"]
    ].mean(axis=1)
    df["temperature_2m_max"] = df[
        ["location_000 temperature_2m", "location_001 temperature_2m"]
    ].max(axis=1)
    df["shortwave_radiation_mean"] = df[
        ["location_000 shortwave_radiation", "location_001 shortwave_radiation"]
    ].mean(axis=1)
    df["cloudcover_mean"] = df[
        ["location_000 cloudcover", "location_001 cloudcover"]
    ].mean(axis=1)
    df["relativehumidity_2m_mean"] = df[
        ["location_000 relativehumidity_2m", "location_001 relativehumidity_2m"]
    ].mean(axis=1)
    df["weathercode_mean"] = df[
        ["location_000 weathercode", "location_001 weathercode"]
    ].mean(axis=1)
    df["effective_radiation"] = df["shortwave_radiation_mean"] * (
        1 - df["cloudcover_mean"] / 100
    )

    return df


def build_daily_dataset(hourly_df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        hourly_df.groupby("Date")
        .agg(
            total_sun_rt=("sun_rt", "sum"),
            max_radiation=("shortwave_radiation_mean", "max"),
            sun_rt_lag_3days_sum=("sun_rt_lag_3days", "sum"),
            temperature_2m_mean=("temperature_2m_mean", "mean"),
            shortwave_radiation_mean=("shortwave_radiation_mean", "mean"),
            cloudcover_mean=("cloudcover_mean", "mean"),
            relativehumidity_2m_mean=("relativehumidity_2m_mean", "mean"),
            weathercode_mean=("weathercode_mean", "mean"),
            effective_radiation=("effective_radiation", "mean"),
        )
        .reset_index()
    )

    daily["trnd"] = range(1, len(daily) + 1)
    daily["w_day"] = daily["Date"].dt.day_name()

    for column in [
        "total_sun_rt",
        "max_radiation",
        "sun_rt_lag_3days_sum",
        "temperature_2m_mean",
        "shortwave_radiation_mean",
        "cloudcover_mean",
        "relativehumidity_2m_mean",
        "weathercode_mean",
        "effective_radiation",
    ]:
        daily[f"log_{column}"] = np.log1p(daily[column].clip(lower=0))

    return daily


def build_model_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    weekday_features = pd.get_dummies(daily_df["w_day"], prefix="w_day")
    features = pd.concat([daily_df[LOG_FEATURES + ["trnd"]], weekday_features], axis=1)
    return features.replace([np.inf, -np.inf], np.nan).astype(float)


def forecast_daily_totals(daily_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    forecast_results = []
    forecast_days = daily_df["Date"].iloc[-11:-1].reset_index(drop=True)

    for current_day in forecast_days:
        target_day = current_day + pd.Timedelta(days=1)
        latest_actual_day = current_day - pd.Timedelta(days=2)

        train_mask = daily_df["Date"] <= latest_actual_day
        valid_train = (
            train_mask
            & feature_df.notna().all(axis=1)
            & daily_df["total_sun_rt"].notna()
            & (daily_df["total_sun_rt"] > 0)
        )

        target_index = daily_df.index[daily_df["Date"] == target_day].tolist()
        if valid_train.sum() < 30 or not target_index:
            forecast_results.append({"Date": target_day, "forecast": np.nan})
            continue

        target_index = target_index[0]
        target_features = feature_df.loc[[target_index]]
        if target_features.isna().any(axis=None):
            forecast_results.append({"Date": target_day, "forecast": np.nan})
            continue

        y_train = np.log1p(daily_df.loc[valid_train, "total_sun_rt"]).astype(float)
        x_train = feature_df.loc[valid_train].astype(float)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                model = SARIMAX(
                    y_train.to_numpy(),
                    exog=x_train.to_numpy(),
                    order=(1, 1, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            forecast_log = model.forecast(steps=1, exog=target_features.to_numpy())[0]
            forecast = max(float(np.expm1(forecast_log)), 0.0)
        except Exception:
            forecast = np.nan

        forecast_results.append({"Date": target_day, "forecast": forecast})

    return pd.DataFrame(forecast_results)


def build_hourly_profile(hourly_df: pd.DataFrame) -> pd.DataFrame:
    actuals = hourly_df.dropna(subset=["sun_rt"]).copy()
    actuals["week"] = actuals["Date"].dt.isocalendar().week
    actuals["year"] = actuals["Date"].dt.isocalendar().year

    daily_totals = actuals.groupby("Date")["sun_rt"].sum().rename("daily_total")
    actuals = actuals.merge(daily_totals, on="Date", how="left")
    actuals = actuals[actuals["daily_total"] > 0]
    actuals["hourly_ratio"] = actuals["sun_rt"] / actuals["daily_total"]

    return (
        actuals.groupby(["year", "week", "Hour"])["hourly_ratio"]
        .mean()
        .reset_index()
    )


def profile_for_date(profile_df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    iso = date.isocalendar()
    profile = profile_df[
        (profile_df["year"] == iso.year) & (profile_df["week"] == iso.week)
    ]

    if profile.empty:
        fallback_iso = (date - pd.Timedelta(days=7)).isocalendar()
        profile = profile_df[
            (profile_df["year"] == fallback_iso.year)
            & (profile_df["week"] == fallback_iso.week)
        ]

    if profile.empty:
        profile = (
            profile_df[profile_df["week"] == iso.week]
            .groupby("Hour", as_index=False)["hourly_ratio"]
            .mean()
        )

    return profile.dropna(subset=["hourly_ratio"]).copy()


def distribute_daily_forecasts(
    daily_forecasts: pd.DataFrame,
    hourly_profile: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, forecast_row in daily_forecasts.iterrows():
        date = pd.to_datetime(forecast_row["Date"])
        total_forecast = forecast_row["forecast"]
        if pd.isna(total_forecast):
            continue

        profile = profile_for_date(hourly_profile, date)
        if profile.empty or profile["hourly_ratio"].sum() == 0:
            ratios = pd.DataFrame({"Hour": range(24), "hourly_ratio": [1 / 24] * 24})
        else:
            ratios = profile[["Hour", "hourly_ratio"]].copy()
            ratios["hourly_ratio"] = ratios["hourly_ratio"] / ratios["hourly_ratio"].sum()

        for _, ratio_row in ratios.iterrows():
            rows.append(
                {
                    "datetime": date + pd.Timedelta(hours=int(ratio_row["Hour"])),
                    "hourly_forecast": total_forecast * ratio_row["hourly_ratio"],
                }
            )

    return pd.DataFrame(rows)


def run_forecast(config: ForecastConfig) -> list[float]:
    hourly_df = build_hourly_dataset(config)
    daily_df = build_daily_dataset(hourly_df)
    feature_df = build_model_features(daily_df)

    daily_forecasts = forecast_daily_totals(daily_df, feature_df)
    hourly_profile = build_hourly_profile(hourly_df)
    hourly_forecasts = distribute_daily_forecasts(daily_forecasts, hourly_profile)

    if hourly_forecasts.empty:
        raise RuntimeError("Could not create an hourly forecast from the available data.")

    return hourly_forecasts["hourly_forecast"].iloc[-24:].round(4).tolist()


def main() -> None:
    config = ForecastConfig(forecast_date=default_forecast_date())
    print(run_forecast(config))


if __name__ == "__main__":
    main()
