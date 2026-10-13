import pandas as pd
import plotly.express as px
import streamlit as st
from gradelens.data import initialize_db, read_grades, read_experiments

st.set_page_config(page_title="GradeLens Learning Analytics", page_icon="📊", layout="wide")
st.title("GradeLens — Learning Analytics")
st.caption("Demo dashboard for answer scores, grading agreement, and review workload.")
initialize_db()
df = read_grades()
if df.empty:
    st.info("No records found. Run `python scripts/create_demo_data.py` first.")
    st.stop()
df["teacher_score"] = pd.to_numeric(df.teacher_score, errors="coerce")
df["predicted_score"] = pd.to_numeric(df.predicted_score, errors="coerce")
df = df.dropna(subset=["predicted_score"])
a, b, c, d = st.columns(4)
a.metric("Answers graded", f"{len(df):,}")
b.metric("Students", df.student_id.nunique())
matched = df[df.teacher_score.notna()]
c.metric("Exact agreement", f"{(matched.teacher_score == matched.predicted_score).mean():.1%}" if len(matched) else "N/A")
d.metric("Average predicted score", f"{df.predicted_score.mean():.2f} / 3")
left, right = st.columns(2)
with left:
    st.plotly_chart(px.histogram(df, x="predicted_score", title="Predicted score distribution"), use_container_width=True)
with right:
    if len(matched):
        st.plotly_chart(px.density_heatmap(matched, x="teacher_score", y="predicted_score", title="Teacher vs model scores"), use_container_width=True)
    else:
        st.info("No teacher labels available.")
st.subheader("Review queue")
review = matched[matched.teacher_score != matched.predicted_score]
st.write(f"{len(review)} answers differ from teacher scores and may need review.")
st.dataframe(review[["student_id", "question_id", "answer_text", "predicted_score", "teacher_score", "confidence"]].head(100),
             use_container_width=True, hide_index=True)
st.subheader("MLflow experiment records")
experiments = read_experiments()
st.dataframe(experiments, use_container_width=True, hide_index=True) if not experiments.empty else st.caption("Training runs are tracked in MLflow under `mlruns/`.")
