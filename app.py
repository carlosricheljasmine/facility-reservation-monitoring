import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone
from modules.data_processing import validate_dataframe, quality_summary, prepare_for_analysis, PREDICTION_FEATURES
from modules.exploratory_analysis import facility_chart, outcome_chart, leadtime_chart
from modules.prediction import predict_rows
from modules.dashboard import outcome_kpis, risk_queue, risk_chart

ROOT=Path(__file__).parent
st.set_page_config(page_title="Campus Facility Reservation Monitoring",page_icon="🏫",layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(ROOT/"models"/"trained_model.joblib")

@st.cache_data
def load_default_data():
    return pd.read_csv(ROOT/"data"/"cleaned_facility_reservations.csv")

bundle=load_model()
st.sidebar.title("Facility Reservation System")
menu=st.sidebar.selectbox("Module",["Home","Dataset","Data Analysis","Prediction","Dashboard","System Design"])
st.sidebar.caption("Model: Decision Tree • v1.0 • Human review required")

if "df" not in st.session_state:
    st.session_state.df=load_default_data()
if "predictions" not in st.session_state:
    st.session_state.predictions=None

df=st.session_state.df

if menu=="Home":
    st.title("Campus Facility Reservation Monitoring")
    st.write("A decision-support prototype for reviewing facility reservation patterns and estimating reservation outcomes.")
    a,b,c=st.columns(3)
    a.metric("Dataset rows",f"{len(df):,}")
    b.metric("Input features",len(bundle["feature_cols"]))
    c.metric("Model accuracy",f"{bundle['metrics']['accuracy']:.1%}")
    st.info("Predictions support authorized staff review. They must not automatically approve or reject a reservation.")
    st.subheader("Workflow")
    st.markdown("1. Load or upload reservation data.<br>2. Validate the schema and inspect data quality.<br>3. Explore observed reservation patterns.<br>4. Enter pre-reservation details or score rows from a CSV.<br>5. Have an authorized administrator review predictions and record the final decision in the official system.", unsafe_allow_html=True)
    st.caption("Prediction checkpoint: when a reservation is submitted, before its eventual usage outcome is known.")

elif menu=="Dataset":
    st.title("Dataset Upload and Validation")
    upload=st.file_uploader("Upload a facility reservation CSV",type=["csv"])
    if upload is not None:
        try:
            candidate=pd.read_csv(upload)
            errs=validate_dataframe(candidate)
            if errs:
                for e in errs: st.error(e)
                st.warning("The uploaded file was not made the active dataset because it did not pass validation.")
            else:
                st.session_state.df=candidate
                df=candidate
                st.success("CSV loaded and schema validation passed.")
        except Exception as exc: st.error(f"Could not read CSV: {exc}")
    errs=validate_dataframe(df)
    if errs:
        st.warning("Current dataset validation notes")
        for e in errs: st.write("• "+e)
    summary=quality_summary(df)
    a,b,c,d=st.columns(4)
    a.metric("Rows",summary["rows"]); b.metric("Columns",summary["columns"]); c.metric("Missing cells",summary["missing_cells"]); d.metric("Duplicate rows",summary["duplicate_rows"])
    st.subheader("Data preview")
    st.dataframe(df.head(30),use_container_width=True)
    st.subheader("Missing values by column")
    st.dataframe(df.isna().sum().rename("Missing values").to_frame(),use_container_width=True)
    st.download_button("Download active dataset",df.to_csv(index=False).encode("utf-8"),"active_facility_reservations.csv","text/csv")

elif menu=="Data Analysis":
    st.title("Exploratory Data Analysis")
    st.caption("Charts summarize observed dataset records; they are not model predictions.")
    num=df.select_dtypes(include="number").columns.tolist()
    if num: st.subheader("Descriptive statistics"); st.dataframe(df[num].describe().T,use_container_width=True)
    if "Reservation_Outcome" in df: st.subheader("Class distribution"); st.dataframe(df["Reservation_Outcome"].value_counts(dropna=False).rename_axis("Outcome").to_frame("Count"))
    st.plotly_chart(facility_chart(df),use_container_width=True)
    st.plotly_chart(outcome_chart(df),use_container_width=True)
    st.plotly_chart(leadtime_chart(df),use_container_width=True)

elif menu=="Prediction":
    st.title("Predictive Modeling")
    st.write("Enter fields available at reservation submission. Do not enter future outcome or attendance information.")
    st.caption("Model inputs: "+", ".join(bundle["feature_cols"]))
    with st.form("prediction_form"):
        cats={c:sorted(df[c].dropna().astype(str).unique().tolist()) for c in ["User_Category","Facility_Type","Time_Slot","Day_of_Week"]}
        c1,c2=st.columns(2)
        user=c1.selectbox("User category",cats["User_Category"])
        facility=c2.selectbox("Facility type",cats["Facility_Type"])
        lead=c1.number_input("Lead time (days)",min_value=0,value=3,step=1)
        participants=c2.number_input("Expected participants",min_value=1,value=10,step=1)
        slot=c1.selectbox("Time slot",cats["Time_Slot"])
        day=c2.selectbox("Day of week",cats["Day_of_Week"])
        submit=st.form_submit_button("Predict outcome")
    if submit:
        row=pd.DataFrame([{"User_Category":user,"Facility_Type":facility,"Lead_Time_Days":lead,"Participants":participants,"Time_Slot":slot,"Day_of_Week":day}])
        try:
            out=predict_rows(row,bundle)
            st.session_state.last_prediction=out
            st.success("Prediction generated. Authorized human review is still required.")
            st.metric("Predicted outcome",out.iloc[0]["Predicted_Outcome"])
            st.metric("Model confidence",f"{out.iloc[0]['Prediction_Confidence']:.1%}")
            st.warning("Confidence is not a guarantee of correctness. Do not automatically reject or approve based on this result.")
        except Exception as exc: st.error(str(exc))
    st.divider()
    st.subheader("Batch prediction from CSV")
    batch=st.file_uploader("Upload CSV with the six model input columns",type=["csv"],key="batch")
    if batch is not None:
        try:
            raw=pd.read_csv(batch)
            if not set(bundle["feature_cols"]).issubset(raw.columns): raise ValueError("CSV must include: "+", ".join(bundle["feature_cols"]))
            scored=predict_rows(raw,bundle); st.session_state.predictions=scored
            st.dataframe(scored,use_container_width=True)
            st.download_button("Download predictions",scored.to_csv(index=False).encode("utf-8"),"reservation_predictions_for_review.csv","text/csv")
        except Exception as exc: st.error(f"Batch prediction failed: {exc}")
    st.subheader("Evaluation metrics from the fixed midterm-style test split")
    m=bundle["metrics"]
    x1,x2,x3,x4=st.columns(4)
    x1.metric("Accuracy",f"{m['accuracy']:.3f}"); x2.metric("Weighted precision",f"{m['precision_weighted']:.3f}"); x3.metric("Weighted recall",f"{m['recall_weighted']:.3f}"); x4.metric("Weighted F1",f"{m['f1_weighted']:.3f}")
    st.write("Confusion matrix (rows = actual class, columns = predicted class; order: "+", ".join(m["class_labels"])+")")
    st.dataframe(pd.DataFrame(m["confusion_matrix"],index=m["class_labels"],columns=m["class_labels"]),use_container_width=True)

elif menu=="Dashboard":
    st.title("Facility Administrator Decision-Support Dashboard")
    st.caption("Observed facts and model predictions are explicitly separated.")
    st.subheader("Observed dataset facts")
    k=outcome_kpis(df)
    a,b,c,d=st.columns(4)
    a.metric("Total reservations",k["Total reservations"]); b.metric("Completed",k["Completed"]); c.metric("Not completed",k["Not completed"]); d.metric("Non-completion rate",f"{k['Non-completion rate']:.1f}%")
    st.subheader("Predictions requiring human review")
    if st.session_state.predictions is not None and not st.session_state.predictions.empty:
        scored=risk_queue(st.session_state.predictions)
        st.plotly_chart(risk_chart(scored),use_container_width=True)
        st.dataframe(scored,use_container_width=True)
    else:
        st.info("No batch predictions are loaded. Use the Prediction module to score a CSV.")
    st.markdown("**Recommended follow-up:** verify reservation details, contact the requester when appropriate, review relevant policy and availability, then let an authorized administrator record the final decision.")

elif menu=="System Design":
    st.title("Systems Analysis and Design Integration")
    st.subheader("Functional requirements")
    for x in ["FR1: The system shall accept CSV uploads and validate required schema.","FR2: The system shall display data preview, missing-value counts, and duplicate summaries.","FR3: The system shall show descriptive statistics and at least three interactive charts.","FR4: The system shall load the saved trained model and predict from six submission-time features.","FR5: The system shall show model metrics, a review queue, interpretations, and human follow-up guidance."]: st.write("• "+x)
    st.subheader("Non-functional requirements")
    for x in ["NFR1 — Usability: clear navigation, labels, and actionable validation errors.","NFR2 — Reliability: cached model loading, fixed model artifact, and repeatable test split.","NFR3 — Privacy/security: use synthetic or authorized data; do not commit sensitive records or secrets."]: st.write("• "+x)
    st.subheader("Actors and use case diagram")
    st.image(str(ROOT/"assets"/"use_case_diagram.png"),caption="Monochrome use case diagram")
    st.subheader("Analytical process flow")
    st.code("Reservation requester submits details → system validates input → saved model estimates outcome → dashboard presents result and limitations → authorized facility administrator reviews evidence → administrator records final decision in official system",language="text")
    st.write("Actors: Reservation Requester (submits details); Facility Administrator (views statistics, reviews predictions, records final decision); System Administrator (manages application and model artifact).")
    st.write("Limitations: training data may be synthetic; category encoding rejects unseen categories; model quality depends on representativeness and data quality; confidence is not calibrated unless separately evaluated.")
