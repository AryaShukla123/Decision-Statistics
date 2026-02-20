import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

st.set_page_config(page_title="Confidence Interval Tool", layout="centered")

st.title("📊 Confidence Interval & Hypothesis Tool")
st.markdown("Analyze your data to find the true mean and test against a target value.")

# --- Sidebar Inputs ---
st.sidebar.header("Input Data")
data_source = st.sidebar.radio("Input Type", ["Manual Summary", "Raw Data (CSV/List)"])

if data_source == "Manual Summary":
    mean = st.sidebar.number_input("Sample Mean (x̄)", value=50.0)
    std_dev = st.sidebar.number_input("Standard Deviation (s)", value=5.0)
    n = st.sidebar.number_input("Sample Size (n)", value=30, min_value=2)
else:
    raw_input = st.sidebar.text_area("Enter numbers separated by commas", "48, 52, 45, 55, 50, 49, 51")
    data = [float(x.strip()) for x in raw_input.split(",")]
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    n = len(data)

confidence = st.sidebar.slider("Confidence Level", 0.80, 0.99, 0.95, step=0.01)

# --- Calculation Logic ---
if n >= 30:
    test_type = "Z-Test"
    critical_value = stats.norm.ppf((1 + confidence) / 2)
    dist_name = "Normal (Z)"
else:
    test_type = "T-Test"
    df = n - 1
    critical_value = stats.t.ppf((1 + confidence) / 2, df)
    dist_name = f"Student's T (df={df})"

standard_error = std_dev / np.sqrt(n)
margin_of_error = critical_value * standard_error
lower_bound = mean - margin_of_error
upper_bound = mean + margin_of_error

# --- Results Display ---
st.info(f"🧬 **Methodology:** Using a **{test_type}** ({dist_name}) based on sample size $n={n}$.")

c1, c2, c3 = st.columns(3)
c1.metric("Sample Mean", f"{mean:.2f}")
c2.metric("Margin of Error", f"±{margin_of_error:.2f}")
c3.metric("Critical Value", f"{critical_value:.3f}")

st.success(f"We are **{confidence*100:.0f}% confident** the true mean is between **{lower_bound:.2f}** and **{upper_bound:.2f}**.")

# --- Hypothesis Testing ---
st.divider()
st.header("🧪 Hypothesis Testing")
null_hypothesis = st.number_input("Null Hypothesis Value (μ₀):", value=mean) # Defaults to mean for clean start

# Logic for P-Value
test_stat = (mean - null_hypothesis) / standard_error
if n >= 30:
    p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
else:
    p_value = 2 * (1 - stats.t.cdf(abs(test_stat), df))

if p_value <= 0.05:
    st.error(f"**Significant Result!** P-value = {p_value:.4f}. Reject H₀.")
else:
    st.warning(f"**Not Significant.** P-value = {p_value:.4f}. Fail to reject H₀.")

# --- Visualization ---
fig = go.Figure()

# 1. Confidence Interval Line (The Whiskers)
fig.add_trace(go.Scatter(
    x=[lower_bound, upper_bound], y=[1, 1],
    mode="lines+markers",
    line=dict(color="royalblue", width=6),
    marker=dict(size=12, symbol="line-ns-open"),
    name=f"{confidence*100}% CI"
))

# 2. Sample Mean Point
fig.add_trace(go.Scatter(
    x=[mean], y=[1],
    mode="markers",
    marker=dict(color="blue", size=14, line=dict(width=2, color="white")),
    name="Sample Mean"
))

# 3. Null Hypothesis Line (The Target)
fig.add_trace(go.Scatter(
    x=[null_hypothesis], y=[1],
    mode="markers",
    marker=dict(color="red", size=18, symbol="x"),
    name="Null Hypothesis (H₀)"
))

fig.update_layout(
    title="Visualizing Uncertainty vs. Hypothesis",
    xaxis_title="Value",
    yaxis=dict(showticklabels=False, range=[0.8, 1.2]),
    height=300,
    showlegend=True
)

st.plotly_chart(fig)