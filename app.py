import streamlit as st
import yfinance as yf
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from portfolio import calculate_portfolio

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Premium Portfolio AI", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}
.title {
    font-size: 48px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Controls")

stocks = st.sidebar.multiselect(
    "Select Stocks",
    ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN"],
    default=["AAPL", "TSLA"]
)

risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Medium", "High"])

run = st.sidebar.button("🚀 Run Analysis")

# -------------------- HEADER --------------------
st.markdown("""
<h1 class="title">🚀 Premium AI Portfolio Dashboard</h1>
<p style='color:gray;'>Smart Investment Insights using Machine Learning</p>
""", unsafe_allow_html=True)

# -------------------- MAIN --------------------
if run:

    if len(stocks) < 2:
        st.warning("Select at least 2 stocks")
        st.stop()

    data = yf.download(stocks, period="1y")["Close"]
    returns = data.pct_change().dropna()

    weights, sharpe = calculate_portfolio(returns, risk_level)

    # -------------------- KPI CARDS --------------------
    st.markdown("## 📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    def metric_card(title, value):
        st.markdown(f"""
        <div class="card">
            <h4>{title}</h4>
            <h2>{value}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        metric_card("📈 Avg Return", f"{returns.mean().mean():.4f}")

    with col2:
        metric_card("⚠️ Risk", f"{returns.std().mean():.4f}")

    with col3:
        metric_card("💼 Sharpe Ratio", f"{sharpe:.4f}")

    st.markdown("---")

    # -------------------- PRICE CHART --------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Price Trends")

    fig = px.line(data, title="Stock Prices")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------- PORTFOLIO PIE --------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💼 Portfolio Allocation")

    fig2 = px.pie(values=weights, names=stocks)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -------------------- RISK VS RETURN --------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📉 Risk vs Return")

    mean = returns.mean()
    risk = returns.std()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=risk,
        y=mean,
        mode='markers+text',
        text=stocks,
        textposition="top center"
    ))

    fig3.update_layout(
        xaxis_title="Risk",
        yaxis_title="Return"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -------------------- PREDICTIONS --------------------
    st.markdown("## 🔮 Predictions (Random Forest)")

    for stock in stocks:
        try:
            model = joblib.load(f"models/{stock}_model.pkl")
            last_price = data[stock].iloc[-1]
            pred = model.predict([[last_price]])[0]

            st.markdown(f"""
            <div class="card">
            <b>{stock}</b><br>
            Predicted Next Price: {pred:.2f}
            </div>
            """, unsafe_allow_html=True)

        except:
            st.error(f"{stock} model missing")

    st.markdown("---")

    # -------------------- RECOMMENDATION --------------------
    st.markdown("## 💡 AI Recommendation")

    if sharpe > 0.1:
        st.success("✅ Strong portfolio. Good risk-return balance.")
    else:
        st.warning("⚠️ Portfolio is weak. Try different stocks or risk level.")

    st.markdown("## 🧠 Model Used")
    st.info("Random Forest Machine Learning Model")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | ML | Portfolio Optimization")