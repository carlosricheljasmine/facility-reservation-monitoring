import plotly.express as px
import pandas as pd

def outcome_kpis(df):
    total=len(df)
    outcome=df["Reservation_Outcome"].astype(str) if "Reservation_Outcome" in df else pd.Series(dtype=str)
    not_completed=int(outcome.eq("Not Completed").sum())
    return {"Total reservations":total,"Completed":int(outcome.eq("Completed").sum()),"Not completed":not_completed,"Non-completion rate":(not_completed/total*100 if total else 0)}

def risk_queue(predictions):
    if predictions is None or predictions.empty: return predictions
    out=predictions.copy()
    out["Review Priority"]=out["Predicted_Outcome"].map(lambda x:"Review suggested" if x=="Not Completed" else "Routine review")
    return out

def risk_chart(predictions):
    counts=predictions["Predicted_Outcome"].value_counts().rename_axis("Predicted outcome").reset_index(name="Count")
    return px.bar(counts,x="Predicted outcome",y="Count",title="Model predictions (not observed outcomes)")
