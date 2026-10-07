import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantlab import OptionType, bs_price

st.title("Black-Scholes")
st.latex(r"dS(t) = r\,S(t)\,dt + \sigma\,S(t)\,dW^{\mathbb{Q}}(t)")
st.caption("Assumptions: constant volatility, constant rate, no jumps, continuous hedging, no frictions.")

with st.sidebar:
    S0 = st.slider("Spot S0", 50.0, 150.0, 100.0)
    T = st.slider("Maturity T (years)", 0.05, 5.0, 1.0)
    r = st.slider("Rate r", 0.0, 0.10, 0.05, step=0.005)
    sigma = st.slider("Volatility sigma", 0.05, 0.80, 0.20, step=0.01)

strikes = np.linspace(0.5 * S0, 1.5 * S0, 101)
calls = [bs_price(OptionType.Call, S0, K, T, r, sigma) for K in strikes]
puts = [bs_price(OptionType.Put, S0, K, T, r, sigma) for K in strikes]

fig = go.Figure()
fig.add_scatter(x=strikes, y=calls, name="Call")
fig.add_scatter(x=strikes, y=puts, name="Put")
fig.update_layout(xaxis_title="Strike K", yaxis_title="Option price", height=450)
st.plotly_chart(fig, width="stretch")
