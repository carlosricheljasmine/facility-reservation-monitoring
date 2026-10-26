import pandas as pd
REQUIRED_COLUMNS = ["Reservation_ID", "User_Category", "Facility_Type", "Lead_Time_Days", "Participants", "Actual_Attendance", "Usage_Status", "Time_Slot", "Day_of_Week", "Lead_Time_Category", "Reservation_Outcome", "Attendance_Ratio"]
PREDICTION_FEATURES = ["User_Category", "Facility_Type", "Lead_Time_Days", "Participants", "Time_Slot", "Day_of_Week"]

def validate_dataframe(df):
    errors=[]
    if df.empty: errors.append("The uploaded CSV has no data rows.")
    missing=[c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: errors.append("Missing required columns: " + ", ".join(missing))
    if "Reservation_ID" in df.columns and df["Reservation_ID"].duplicated().any():
        errors.append(f"Duplicate Reservation_ID values: {int(df['Reservation_ID'].duplicated().sum())}")
    return errors

def quality_summary(df):
    return {"rows":len(df),"columns":len(df.columns),"missing_cells":int(df.isna().sum().sum()),"duplicate_rows":int(df.duplicated().sum()),"duplicate_reservation_ids":int(df["Reservation_ID"].duplicated().sum()) if "Reservation_ID" in df.columns else None}

def prepare_for_analysis(df):
    out=df.copy()
    for c in ["Lead_Time_Days","Participants","Actual_Attendance","Attendance_Ratio"]:
        if c in out.columns: out[c]=pd.to_numeric(out[c],errors="coerce")
    return out
