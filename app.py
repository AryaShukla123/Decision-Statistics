import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

st.set_page_config(page_title="Confidence Interval Tool", layout="centered")

st.title("📊 Confidence Interval Calculator")
st.markdown("Determine the range where the **true population mean** likely lies.")

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

# 1. Determine the Test Type
if n >= 30:
    test_type = "Z-Test"
    # Use Normal Distribution (Z)
    critical_value = stats.norm.ppf((1 + confidence) / 2)
    dist_name = "Normal (Z)"
else:
    test_type = "T-Test"
    # Use Student's T-Distribution (T)
    df = n - 1
    critical_value = stats.t.ppf((1 + confidence) / 2, df)
    dist_name = f"Student's T (df={df})"

# 2. Standard Error (remains the same)
standard_error = std_dev / np.sqrt(n)

# 3. Margin of Error
margin_of_error = critical_value * standard_error

# 4. The Interval
lower_bound = mean - margin_of_error
upper_bound = mean + margin_of_error

# --- Smart UI Feedback ---
st.info(f"🧬 **Methodology:** Automatically selected a **{test_type}** because your sample size is **{n}**. Using critical value from {dist_name} distribution.")

# Update the metrics display
col1, col2, col3 = st.columns(3)
col1.metric("Mean", round(mean, 2))
col2.metric(f"{test_type} Score", round(critical_value, 3))
col3.metric("Margin of Error", f"±{round(margin_of_error, 2)}")

# --- Display Results ---
col1, col2, col3 = st.columns(3)
col1.metric("Mean", round(mean, 2))
col2.metric("Margin of Error", f"±{round(margin_of_error, 2)}")
col3.metric("Sample Size", n)

st.success(f"We are **{confidence*100:.0f}% confident** that the true mean is between **{lower_bound:.2f}** and **{upper_bound:.2f}**.")

# --- Visualization ---
fig = go.Figure()

# Add the Interval Line
fig.add_trace(go.Scatter(
    x=[lower_bound, upper_bound], y=[1, 1],
    mode="lines+markers",
    line=dict(color="royalblue", width=4),
    marker=dict(size=12, symbol="line-ns-open"),
    name="Confidence Interval"
))

# Add the Point Estimate (Mean)
fig.add_trace(go.Scatter(
    x=[mean], y=[1],
    mode="markers",
    marker=dict(color="firebrick", size=15),
    name="Sample Mean"
))

fig.update_layout(
    title=f"{confidence*100}% Confidence Interval Visual",
    xaxis_title="Value Range",
    yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
    height=300
)

st.plotly_chart(fig)