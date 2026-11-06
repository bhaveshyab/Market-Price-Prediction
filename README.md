# Farm Price Outlook

A small, reproducible Streamlit application for exploring Indian agricultural market observations and evaluating a multi-crop price model. The forecast target is the **exact modal price seven calendar days after the observation**, for the same state, district, market, commodity, variety, and grade. A missing exact date stays missing; the pipeline does not replace it with the next reported day.

## Data and source assumptions

Place source CSV/XLSX files in the project root or a `data/raw` folder. Files need State, District, Market, Commodity, Variety, Grade, Arrival_Date, and Modal_Price columns (case and common separators are normalized). Optional minimum and maximum prices are retained and checked. Invalid dates and nonpositive modal prices are excluded from modeling; zeros and negatives are counted in the report. Exact duplicate market-series/date observations keep the first record. Price range relationship violations are reported but kept. Original source files are never rewritten.

The project includes a historical Tomato CSV and the full 2025 workbook (`2025.csv.xlsx`). The raw workbook was loaded and processed in full. `2025 cleaned.xlsx` is preserved as supplied. The generated `artifacts/commodity_coverage.csv` lists every commodity and its record, series, exact-seven-day coverage, split counts, and forecast support status.

## Windows PowerShell setup

From this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Prepare data, train, evaluate, and save outputs under the ignored `artifacts` directory:

```powershell
python .\train_model.py
```

Train and evaluate the checked-in Tomato CSV (the completed, limited real-data run described below):

```powershell
python .\train_model.py --data-file .\tomato_clean.csv
```

Run the full source workbook pipeline from a separate data folder or choose another output folder:

```powershell
python .\train_model.py --data-dir .\data\raw --output-dir .\artifacts
```

The dashboard loads the multi-commodity seven-day model and cleaned data from `artifacts/`. Filters start with commodity and cascade through state, district, market, variety, and grade. A forecast is disabled when the commodity has fewer than 20 exact-seven-day training targets or the selected series has fewer than 8 observations. The older `train_multi_crop.py` is a separate next-observation experiment and is not used by the dashboard.

Run unit tests and open the dashboard:

```powershell
python -m pytest -q
python -m streamlit run .\app.py
```

## Evaluation design

Targets are joined by the exact date `observation + 7 days` inside each full series key. Inputs include dimensions, calendar fields, and lags/rolling values shifted to use prior observations only. Split boundaries use observation dates in chronological order (70% train, 15% validation, 15% test). The saved estimator is XGBoost when installed, with a Random Forest fallback. Persistence predicts the latest previous observed modal price. MAE, RMSE, R-squared, date ranges, sample counts, exact-target coverage, predictions, model, and feature importance (when supported) are written to `artifacts/`. R-squared may be negative.

The validation partition is reserved chronologically but the model is not tuned against it. Exact seven-day target coverage can be low for markets that report intermittently; training stops clearly if any split has no target examples. A forecast is an estimate, not market advice. Source units are displayed without asserting currency or weight because workbook units have not yet been verified.

## Full 2025 run

The full `2025.csv.xlsx` sheet loaded 1,048,575 data rows. Cleaning retained 1,048,382 rows, removed 193 invalid rows, and found no duplicate series/date records. The retained dates span 2025-01-01 through 2025-02-23 (54 distinct dates). The data contains 297 commodities and 2,677 market names. The model's exact seven-day target and historical-feature requirements yielded 661,604 eligible examples (63.1% of retained rows). The chronological split has 436,382 training, 113,170 validation, and 112,052 test examples.

| Test metric | Current-price persistence | XGBoost |
| --- | ---: | ---: |
| MAE | 441.55 | 438.32 |
| RMSE | 3,072.74 | 2,627.68 |
| R-squared | 0.7718 | 0.8332 |

Of the 297 commodities, 196 have at least 20 exact-seven-day training targets and are forecast-enabled; 101 are clearly marked insufficient in the dashboard and coverage CSV. The full per-commodity record and target coverage is in `artifacts/commodity_coverage.csv`. These results use the full cleaned workbook data. The original Excel files were not modified.

