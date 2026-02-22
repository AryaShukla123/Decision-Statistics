import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

st.set_page_config(page_title="Decision-Statistics", layout="centered")

st.title("📊 Decision-Statistics")
st.markdown("Automated Statistical Inference & Relationship Analysis")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
analysis_mode = st.sidebar.selectbox("Select Analysis Mode", ["Univariate (One Variable)", "Bivariate (Relationship)"])

# --- MODE 1: UNIVARIATE ---
if analysis_mode == "Univariate (One Variable)":
    st.sidebar.subheader("Input Data")
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
    null_hypothesis = st.number_input("Null Hypothesis Value (μ₀):", value=mean)
    test_stat = (mean - null_hypothesis) / standard_error
    p_value = 2 * (1 - stats.norm.cdf(abs(test_stat))) if n >= 30 else 2 * (1 - stats.t.cdf(abs(test_stat), df))

    if p_value <= 0.05:
        st.error(f"**Significant Result!** P-value = {p_value:.4f}. Reject H₀.")
    else:
        st.warning(f"**Not Significant.** P-value = {p_value:.4f}. Fail to reject H₀.")

    # --- Visualization (Univariate) ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lower_bound, upper_bound], y=[1, 1], mode="lines+markers", line=dict(color="royalblue", width=6), name=f"{confidence*100}% CI"))
    fig.add_trace(go.Scatter(x=[mean], y=[1], mode="markers", marker=dict(color="blue", size=14), name="Sample Mean"))
    fig.add_trace(go.Scatter(x=[null_hypothesis], y=[1], mode="markers", marker=dict(color="red", size=18, symbol="x"), name="Null Hypothesis (H₀)"))
    fig.update_layout(title="Visualizing Uncertainty vs. Hypothesis", yaxis=dict(showticklabels=False, range=[0.8, 1.2]), height=300)
    st.plotly_chart(fig)

    # --- N-Optimizer ---
    st.divider()
    st.header("🎯 N-Optimizer")
    target_moe = st.slider("Target Margin of Error", min_value=float(margin_of_error * 0.1), max_value=float(margin_of_error * 1.5), value=float(margin_of_error * 0.8))
    required_n = np.ceil((critical_value * std_dev / target_moe) ** 2)
    st.write(f"Total sample size needed for ±{target_moe:.2f}:")
    st.title(f"n = {int(required_n)}")

# --- MODE 2: BIVARIATE ---
elif analysis_mode == "Bivariate (Relationship)":
    st.header("📈 Linear Regression & Correlation")
    st.write("Upload two sets of data to see if they are related.")

    col_a, col_b = st.columns(2)
    with col_a:
        x_input = st.text_area("Independent Variable (X)", "1, 2, 3, 4, 5")
    with col_b:
        y_input = st.text_area("Dependent Variable (Y)", "2, 4, 5, 4, 5")

    x = [float(i.strip()) for i in x_input.split(",")]
    y = [float(i.strip()) for i in y_input.split(",")]

    if len(x) == len(y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value**2

        m1, m2, m3 = st.columns(3)
        m1.metric("Correlation (r)", round(r_value, 3))
        m2.metric("R-Squared", f"{r_squared:.2%}")
        m3.metric("P-Value", round(p_value, 4))

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Points'))
        line = [slope * i + intercept for i in x]
        fig.add_trace(go.Scatter(x=x, y=line, mode='lines', name='Regression Line', line=dict(color='red')))
        fig.update_layout(title=f"Equation: y = {slope:.2f}x + {intercept:.2f}", xaxis_title="X", yaxis_title="Y")
        st.plotly_chart(fig)

        # Prediction
        st.subheader("🔮 Make a Prediction")
        predict_x = st.number_input(f"Enter a value for X to predict Y:", value=float(max(x)))
        prediction = slope * predict_x + intercept
        st.success(f"Predicted Y value: **{prediction:.2f}**")
    else:
        st.error("Error: X and Y must have the same number of data points.")