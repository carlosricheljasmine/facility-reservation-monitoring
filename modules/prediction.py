import pandas as pd

def encode_features(raw_df, bundle):
    features=bundle["feature_cols"]
    missing=[c for c in features if c not in raw_df.columns]
    if missing: raise ValueError("Missing model features: " + ", ".join(missing))
    X=raw_df[features].copy()
    for col, encoder in bundle["encoders"].items():
        vals=X[col].fillna("Unknown").astype(str).str.strip()
        unknown=sorted(set(vals)-set(encoder.classes_))
        if unknown: raise ValueError(f"Unseen category for {col}: {', '.join(unknown)}. Choose a category present in the training dataset.")
        X[col]=encoder.transform(vals)
    for col in ["Lead_Time_Days","Participants"]:
        X[col]=pd.to_numeric(X[col],errors="coerce")
        if X[col].isna().any(): raise ValueError(f"{col} must be a valid number.")
    return X

def predict_rows(raw_df,bundle):
    X=encode_features(raw_df,bundle)
    model=bundle["model"]
    encoded=model.predict(X)
    labels=bundle["target_encoder"].inverse_transform(encoded)
    result=raw_df.copy()
    result["Predicted_Outcome"]=labels
    if hasattr(model,"predict_proba"):
        result["Prediction_Confidence"]=model.predict_proba(X).max(axis=1)
    result["Human_Review_Required"]=True
    return result
