# Final Report — AT2: Machine Learning as a Service (Summary)

**Student / Client:** Replace with your name and student id  
**Project:** Predictive Weather Models for Sydney (classification + regression)

## 1. Objectives
- Build two models for Sydney (lat -33.8678, lon 151.2073):
  1. Binary classification: will it rain exactly +7 days from an input date (rain = precipitation > 0).
  2. Regression: cumulative precipitation (mm) in the next 3 days from an input date.

## 2. Data
- Source: Open-Meteo historical API (archive endpoint).
- Frequency: daily aggregated values (precipitation_sum, rain_sum, temperature_2m_mean, etc.).
- Training window: All historical data up to end of 2024.
- Production data: 2025 onwards (kept separate and NOT used in training).

## 3. Feature engineering
- Lag features: precipitation mean & sum for 1, 3, 7, 14 day windows.
- Temperature aggregates (7-day mean/std) if available.
- Day-of-year cyclical features (sin/cos) to capture seasonality.
- Yesterday rain indicator (binary).
- Labels:
  - Classification: precipitation at input_date + 7 > 0 -> 1 else 0.
  - Regression: cumulative precipitation input_date+1 .. input_date+3.

## 4. Models & evaluation
- Baseline approach (provided scripts):
  - RandomForestClassifier for classification.
  - RandomForestRegressor for regression.
- Suggested metrics:
  - Classification: Accuracy, Precision, Recall, F1, ROC-AUC.
  - Regression: MAE, RMSE, R^2.
- Data splits: train ≤ 2021-12-31, val: 2022-2023, test: 2024.

## 5. Implementation notes
- Reproducible scripts included in `src/`:
  - `data_fetch.py`, `fetch_all_years.py`, `merge_daily.py`, `features.py`, `build_features_table.py`, `train.py`.
- Training outputs:
  - Models saved to `models/` as `rain_class_baseline.joblib` and `precip_reg_baseline.joblib`.
  - Feature column metadata saved to `models/feature_columns.json`.
  - Metrics saved to `reports/metrics.json`.

## 6. API
- The FastAPI app (app/main.py) exposes:
  - `/` : overview (project + endpoints + github link).
  - `/health/` : status check.
  - `/predict/rain/?date=YYYY-MM-DD` : returns JSON with input_date and prediction.date = input_date+7 and will_rain boolean.
  - `/predict/precipitation/fall?date=YYYY-MM-DD` : returns JSON with input_date and prediction.start_date, prediction.end_date_date, and precipitation_fall (string).

*Important*: The included API code provides deterministic placeholder predictions if trained models are not placed in `models/`. After training, copy the joblib artifacts to the API `models/` folder for real inference.

## 7. Deployment to Render (high-level)
1. Create a free Render account and a new Web Service.
2. Connect your private GitHub repo to Render (api repo).
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
5. Add models to the repo (or provide a secure storage mechanism and download at runtime).

## 8. Ethics, privacy, and limitations
- Weather data used is public; no personal data is involved.
- Be cautious with high-stakes decisions (flood warnings, safety-critical systems). This model is a forecasting aid — not an authoritative warning system.
- Document seasonal biases, data gaps, and potential model drift. Retrain periodically as new data arrives.

## 9. What I included in this submission
- `experimentation_repo/` containing scripts, placeholder notebooks, and instructions for experiments and training.
- `api_repo/` containing a FastAPI template that matches the assignment endpoints and example Dockerfile.
- `final_report.md` (this document), `README.md` files for both repos, and `github.txt` placeholders.

## 10. Next steps you (the student) must do before submission
1. Replace placeholders (student id, GitHub repo links) in `github.txt`.
2. Run the fetch & training scripts locally (ensure you have API access and rate-limits respected).
3. Save the trained models into the API `models/` folder.
4. Push each repo to **private** GitHub repositories and grant admin access to the instructors' emails listed in the brief.
5. Deploy the API repo to Render and verify endpoints.
6. Zip each repo separately if Canvas requires two zip files (both are included inside the single zip provided).

---

If you want I can:
- Train baseline models here (if you allow me to fetch data from Open-Meteo during this session) and include the trained joblib artifacts in the zip, OR
- Produce a polished PDF version of this final report.

