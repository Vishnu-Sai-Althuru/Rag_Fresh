import streamlit as st
from metrics import get_metrics

st.title("AI Observability Dashboard")

data = get_metrics()

st.metric("Total Requests", data["requests"])
st.metric("Total Tokens", data["tokens"])
st.metric("Errors", data["errors"])

if data["latencies"]:
    st.line_chart(data["latencies"])