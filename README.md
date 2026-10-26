# Campus Facility Reservation Monitoring — Streamlit

A Section A prototype for the Systems Analysis and Design laboratory. It reuses the midterm's six submission-time features, `Reservation_Outcome` target, `LabelEncoder` treatment of categorical inputs, and `DecisionTreeClassifier(max_depth=4, random_state=42)` approach. The saved model is `models/trained_model.joblib`; it is loaded once with Streamlit resource caching.

## Files
- `app.py` — navigation and six modules/pages
- `data/cleaned_facility_reservations.csv` — provided cleaned dataset used by the project
- `modules/` — validation, EDA, prediction, dashboard logic
- `models/trained_model.joblib` — fitted classifier, encoders, and evaluation metrics
- `assets/use_case_diagram.png` — monochrome SAD use case diagram
- `sample_predictions.csv` — five sample predictions from the trained artifact
- `deployment_report.md` — tests, changes, limitations, reflection

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Create a GitHub repository and upload all files while preserving folder structure.
2. Open the official Streamlit Community Cloud site: https://share.streamlit.io
3. Create an app, choose the repository and branch, set `app.py` as the entry point, and deploy.
4. Test the deployed URL and check app logs if the build fails.

## Model notes
The model predicts `Reservation_Outcome` (`Completed` vs `Not Completed`) from `User_Category`, `Facility_Type`, `Lead_Time_Days`, `Participants`, `Time_Slot`, and `Day_of_Week`. Post-outcome fields such as `Usage_Status`, `Actual_Attendance`, `Attendance_Ratio`, and `Reservation_ID` are excluded from inference. Predictions are decision support only; never automatically reject a reservation.
