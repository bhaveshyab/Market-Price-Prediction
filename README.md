# Farm Price Outlook

A small, reproducible Streamlit application for exploring Indian agricultural market observations and evaluating a multi-crop price model. The forecast target is the **exact modal price seven calendar days after the observation**, for the same state, district, market, commodity, variety, and grade. A missing exact date stays missing; the pipeline does not replace it with the next reported day.

## Data and source assumptions

Place source CSV/XLSX files in the project root or a `data/raw` folder. Files need State, District, Market, Commodity, Variety, Grade, Arrival_Date, and Modal_Price columns (case and common separators are normalized). Optional minimum and maximum prices are retained and checked. Invalid dates and nonpositive modal prices are excluded from modeling; zeros and negatives are counted in the report. Exact duplicate market-series/date observations keep the first record. Price range relationship violations are reported but kept. Original source files are never rewritten.

The project contains a historical Tomato CSV (`tomato_clean.csv`, 39,114 usable rows) and two 2025 Excel workbooks. The raw workbook `2025.csv.xlsx` has a sheet named `2025`; its worksheet reports a 1,048,576 by 14 range, with 12 named columns matching the expected market-price schema and two blank header columns. The first two data rows include Gur and Garlic, confirming that the sample is not Tomato-only. The sheet reaches Excel's maximum row index, however, so that value is not a verified nonblank record count. A full import attempt took over 20 minutes without completing in this environment. Full-workbook row counts, commodity counts, date coverage, cleaning report, and model metrics therefore remain unverified. When raw and cleaned files are co-located, training prefers the raw-named files to avoid loading a cleaned duplicate. The file `2025 cleaned.xlsx` already has uncommitted user changes; it was left intact.

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

An additional CLI, `train_multi_crop.py`, is retained from the upstream branch as an explicitly different target experiment. It predicts the **next available recorded observation**, not a fixed seven-day value, and writes its separate reports to `outputs/`. Its model is not loaded by the seven-day dashboard. Compare the target definitions before using either result.

Run unit tests and open the dashboard:

```powershell
python -m pytest -q
python -m streamlit run .\app.py
```

## Evaluation design

Targets are joined by the exact date `observation + 7 days` inside each full series key. Inputs include dimensions, calendar fields, and lags/rolling values shifted to use prior observations only. Split boundaries use observation dates in chronological order (70% train, 15% validation, 15% test). The saved estimator is XGBoost when installed, with a Random Forest fallback. Persistence predicts the latest previous observed modal price. MAE, RMSE, R-squared, date ranges, sample counts, exact-target coverage, predictions, model, and feature importance (when supported) are written to `artifacts/`. R-squared may be negative.

The validation partition is reserved chronologically but the model is not tuned against it. Exact seven-day target coverage can be low for markets that report intermittently; training stops clearly if any split has no target examples. A forecast is an estimate, not market advice. Source units are displayed without asserting currency or weight because workbook units have not yet been verified.

## Current verification status

The new automated tests passed: **7 passed**. Python `compileall` passed. Streamlit served HTTP 200 and an AppTest run rendered three tabs with no app exceptions.

The completed real-data training run used only `tomato_clean.csv` because the 2025 workbook did not finish loading within a practical runtime. It covered 39,114 rows, one commodity (Tomato), 1,028 markets, and dates 2025-01-01 through 2025-02-23. There were no invalid dates/prices or duplicate series-date records; one minimum price was zero. Exact seven-day targets plus required historical features were available for 24,786 rows (63.4%); 14,328 rows were excluded. The chronological split used 16,093 training rows (2025-01-04 through 2025-02-02), 4,387 validation rows, and 4,306 test rows (observation dates 2025-02-10 through 2025-02-16; their target dates are seven days later).

| Test metric | Persistence baseline | Random Forest |
| --- | ---: | ---: |
| MAE | 206.94 | 237.17 |
| RMSE | 376.76 | 369.09 |
| R-squared | 0.8793 | 0.8841 |

The Random Forest has worse MAE than persistence and slightly better RMSE and R-squared. It does not consistently outperform the baseline. XGBoost was not installed in the active environment, so the fallback Random Forest was used. No per-market result met the 20-test-row reporting threshold. These metrics describe the Tomato CSV only and must not be represented as evaluation of the multi-crop workbook. The dashboard currently loads the completed Tomato artifacts; rerun training against a validated multi-crop source to populate it with broader coverage. Do not treat the repository's older saved Tomato model files or their evaluation CSVs as results from this seven-day pipeline.

