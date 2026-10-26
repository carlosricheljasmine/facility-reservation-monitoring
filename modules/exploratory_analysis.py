import plotly.express as px

def facility_chart(df):
    counts=df["Facility_Type"].fillna("Unknown").value_counts().rename_axis("Facility Type").reset_index(name="Reservations")
    return px.bar(counts,x="Facility Type",y="Reservations",title="Reservations by Facility Type")

def outcome_chart(df):
    counts=df["Reservation_Outcome"].fillna("Unknown").value_counts().rename_axis("Outcome").reset_index(name="Reservations")
    return px.pie(counts,names="Outcome",values="Reservations",title="Reservation Outcome Distribution",hole=.35)

def leadtime_chart(df):
    if "Lead_Time_Category" not in df.columns:
        return px.bar(title="Lead-time category is unavailable in this dataset")
    counts=df.groupby(["Lead_Time_Category","Reservation_Outcome"],dropna=False).size().reset_index(name="Reservations")
    return px.bar(counts,x="Lead_Time_Category",y="Reservations",color="Reservation_Outcome",barmode="group",title="Outcome by Lead-Time Category")
