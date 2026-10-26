# Short Deployment Report

## Reused midterm work
The project uses the provided cleaned facility reservation CSV and the midterm notebook's target (`Reservation_Outcome`), six predictor columns (`User_Category`, `Facility_Type`, `Lead_Time_Days`, `Participants`, `Time_Slot`, `Day_of_Week`), categorical `LabelEncoder` approach, stratified 80/20 split with random state 42, and `DecisionTreeClassifier(max_depth=4, random_state=42)`. The model is saved as a joblib artifact and is not retrained during page refreshes.

## Changes made
- Added Streamlit navigation for Home, Dataset, Data Analysis, Prediction, Dashboard, and System Design.
- Added CSV schema validation, preview, missing/duplicate summaries, descriptive statistics, and three interactive Plotly charts.
- Added single-record and batch predictions using only features available at reservation submission.
- Added evaluation metrics, confusion matrix, administrator review queue, functional/non-functional requirements, use case diagram, and analytical workflow.

## Test and model evaluation
- Dataset rows used for modeling: 400
- Training rows: 320; test rows: 80; random state: 42
- Accuracy: 0.5500
- Weighted precision: 0.5267
- Weighted recall: 0.5500
- Weighted F1: 0.5214
- Confusion matrix class order: Completed, Not Completed
- Confusion matrix: [[35, 11], [25, 9]]

## Verification performed
- Confirmed the provided CSV has the columns expected by the project.
- Built the saved model artifact from the same six model features and fixed split pattern described in the notebook.
- Generated five sample predictions and saved them to `sample_predictions.csv`.
- Prediction code checks required features and rejects unknown categorical labels with a readable message.

Before submission, run `streamlit run app.py`, test upload errors, empty files, prediction form, batch prediction, and all pages, then capture screenshots of the running app. A live deployment URL cannot be created from this local project alone; you must push the repository and deploy it using your own account.

## Limitations and reflection
The notebook creates synthetic data if its original source CSV is missing; therefore the provided dataset may be synthetic and results should not be treated as validated real-world performance. The model is a simple decision tree, category encoders cannot handle unseen values without a policy, and the displayed confidence is not necessarily calibrated. Poor or unrepresentative data can produce misleading patterns and predictions. An administrator must check the context and make the final decision; predictions must never automatically reject a reservation.
