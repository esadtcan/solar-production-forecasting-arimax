# Project Report

The final project report is maintained as a LaTeX document:

- Source: [`report/REPORT.tex`](report/REPORT.tex)
- PDF: [`REPORT.pdf`](REPORT.pdf)
- Figures: [`report/figures/`](report/figures/)

To rebuild the PDF locally:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/REPORT.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/REPORT.tex
cp report/REPORT.pdf REPORT.pdf
```

The report includes the problem definition, data coverage, feature engineering,
missing-value analysis, exploratory plots, validation metrics, backtest figures,
operational pipeline, limitations, and references.
