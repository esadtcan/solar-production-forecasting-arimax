# IE360 Solar Production Forecasting with ARIMAX

This repository contains an IE360 course project for forecasting next-day hourly
solar energy production. The model combines historical production, weather
forecasts, calendar effects, and an ARIMAX-style SARIMAX model with exogenous
weather variables.

The main script is [`forecast.py`](forecast.py). It fetches the latest available
data, builds daily production forecasts, distributes the daily forecast into 24
hourly values, and prints the final result as a Python list.

## Project Goal

The goal is to forecast the next day's hourly solar production (`sun_rt`) for a
solar plant using:

- historical solar production data,
- historical and future weather data,
- lagged production information,
- weather-derived explanatory variables,
- day-of-week and trend effects,
- a SARIMAX model with exogenous regressors.

## Repository Structure

```text
.
├── forecast.py          # Main forecasting pipeline
├── forecast_helper.py   # Data fetching utilities
├── README.md            # Project overview and usage guide
└── REPORT.md            # Detailed project report
```

## Data Sources

The project uses two external data sources:

- Production data: a Google Drive CSV file loaded in `get_production_data()`.
- Weather data: Open-Meteo APIs loaded through `openmeteo_requests`.

Weather variables are fetched for two plant coordinates:

- `temperature_2m`
- `shortwave_radiation`
- `cloudcover`
- `relativehumidity_2m`
- `weathercode`

The script combines historical forecast weather data with current forecast data.
For overlapping timestamps, historical values are prioritized.

## Forecasting Pipeline

At a high level, `forecast.py` performs these steps:

1. Define the forecast date. By default, the script forecasts tomorrow.
2. Fetch production data and weather data.
3. Build an hourly table from `2022-01-01` through the forecast day.
4. Create weather features by averaging the two plant locations.
5. Aggregate the hourly table into daily features.
6. Apply log transformations to production and weather variables.
7. Train SARIMAX models using only data that would have been available by `t-2`.
8. Forecast daily solar production.
9. Build historical hourly production profiles by year, ISO week, and hour.
10. Distribute the daily forecast into 24 hourly values.
11. Print the final 24-hour forecast list.

## Installation

This project expects Python 3.9+ and the following packages:

```bash
pip install pandas numpy statsmodels openmeteo-requests requests-cache retry-requests
```

## Usage

Run the main script from the repository root:

```bash
python3 forecast.py
```

The output is a list of 24 numbers, one forecast value for each hour of the
forecast day:

```text
[0.0129, 0.0129, 0.0129, ..., 0.0105]
```

The script requires internet access because it downloads production and weather
data at runtime.

## Main Design Choices

- Daily-first modeling: The model forecasts total daily production first, then
  distributes that total into hourly values. This keeps the SARIMAX model more
  stable than fitting directly on noisy hourly data.
- Weather exogenous variables: Solar production is strongly related to radiation,
  cloud cover, humidity, and temperature, so these are included as model inputs.
- Lagged production: A 3-day lag is used because the most recent actual
  production values may not be available at forecast time.
- Hourly profile distribution: Historical hourly production ratios are used to
  convert daily forecasts into hourly forecasts.

## Limitations

- The model depends on external APIs and a hosted production CSV.
- There is no local test dataset committed to the repository.
- SARIMAX fitting can be sensitive to noisy or missing data.
- The current version does not include automated backtesting output.
- The helper module still contains duplicated weather-fetching logic that could
  be refactored further.

## Future Improvements

- Add a `requirements.txt` file for reproducible installation.
- Add local sample data for offline testing.
- Add a backtesting script with WMAPE or MAE reporting.
- Compare SARIMAX with simpler baselines such as persistence, linear regression,
  and gradient boosting.
- Refactor `forecast_helper.py` to reduce duplicated API code.
