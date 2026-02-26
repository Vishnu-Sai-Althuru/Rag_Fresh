import streamlit as st
import json

st.title("AI Observability Dashboard")

with open("metrics.json") as f:
    metrics = json.load(f)

st.metric("Total Requests", metrics["requests"])
st.metric("Total Tokens", metrics["tokens"])
st.metric("Errors", metrics["errors"])

st.subheader("Latency Trend")

st.line_chart(metrics["latencies"])