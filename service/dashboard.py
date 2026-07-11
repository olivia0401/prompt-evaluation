"""
Streamlit dashboard over the evaluation API.

    streamlit run service/dashboard.py

Set API_BASE_URL (default http://localhost:8000) to point at the FastAPI app.
It talks to the API over HTTP only — no direct DB access — so it works against
a local or a deployed service unchanged.
"""
import os

import httpx
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 30.0

st.set_page_config(page_title="Prompt Eval Dashboard", layout="wide")


def api_get(path: str, **params):
    r = httpx.get(f"{API_BASE_URL}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json=None):
    r = httpx.post(f"{API_BASE_URL}{path}", json=json, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


st.title("Prompt Evaluation Dashboard")

# --- Sidebar: connection + submit ----------------------------------------
with st.sidebar:
    st.header("Connection")
    st.caption(f"API: `{API_BASE_URL}`")
    try:
        health = api_get("/health")
        st.success(f"queue: {health['queue']} · db: {health['database']}")
    except Exception as e:
        st.error(f"API unreachable: {e}")
        st.stop()

    st.header("Submit a run")
    stages = [s["stage"] for s in api_get("/stages")["stages"]]
    with st.form("submit"):
        stage = st.selectbox("Stage", stages)
        budget = st.number_input("Budget (USD, 0 = default)", min_value=0.0, value=0.0, step=0.5)
        max_calls = st.number_input("Max calls (0 = unlimited)", min_value=0, value=0, step=10)
        note = st.text_input("Note", "")
        if st.form_submit_button("Submit run"):
            payload = {"stage": stage, "note": note or None}
            if budget > 0:
                payload["budget_usd"] = budget
            if max_calls > 0:
                payload["max_calls"] = int(max_calls)
            run = api_post("/runs", json=payload)
            st.success(f"Submitted run {run['id']} ({run['status']})")

    if st.button("Refresh"):
        st.rerun()

# --- Runs table -----------------------------------------------------------
runs = api_get("/runs", limit=100)
if not runs:
    st.info("No runs yet. Submit one from the sidebar.")
    st.stop()

df = pd.DataFrame(runs)
show_cols = [c for c in ["id", "stage", "status", "call_count", "planned_calls",
                         "total_cost_usd", "created_at", "finished_at", "note"] if c in df.columns]
st.subheader("Runs")
st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

# --- Run detail -----------------------------------------------------------
run_id = st.selectbox("Inspect run", [r["id"] for r in runs])
if run_id:
    run = api_get(f"/runs/{run_id}")
    metrics = api_get(f"/runs/{run_id}/metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", run["status"])
    c2.metric("Calls", metrics["call_count"])
    c3.metric("Cost (USD)", f"${metrics['total_cost_usd']:.4f}")
    c4.metric("OK rate", f"{(metrics['ok_rate'] or 0) * 100:.1f}%")
    if run.get("error"):
        st.error(run["error"])

    mc1, mc2 = st.columns(2)
    with mc1:
        st.caption("Cost by model")
        if metrics["cost_by_model"]:
            st.bar_chart(pd.Series(metrics["cost_by_model"], name="usd"))
    with mc2:
        st.caption("Status counts")
        if metrics["status_counts"]:
            st.bar_chart(pd.Series(metrics["status_counts"], name="calls"))

    st.caption("Results")
    results = api_get(f"/runs/{run_id}/results", limit=1000)["results"]
    if results:
        rdf = pd.DataFrame(results)
        cols = [c for c in ["brief_id", "task", "config_id", "model_key", "run_index",
                            "status", "parsed_output", "input_tokens", "output_tokens",
                            "cost_usd", "latency_s"] if c in rdf.columns]
        st.dataframe(rdf[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No results recorded yet.")
