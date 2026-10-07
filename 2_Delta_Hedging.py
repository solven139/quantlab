import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantlab import OptionType, bs_delta, bs_price, delta_hedge_pnl, simulate_gbm

CALL_COLOR = "#3987e5"
ACCENT = "#d95926"
NEUTRAL = "#8a8985"

st.title("Delta Hedging")
st.markdown(
    "You are the bank. You **sell** one call at its Black-Scholes price, buy Δ shares, "
    "and rebalance on a time grid until maturity, borrowing or lending through a bank "
    "account at rate r (book Ch 3.3, eqs. 3.42–3.43)."
)
st.latex(r"P\&L(t_i) = P\&L(t_{i-1})\,e^{r\Delta t} - \big(\Delta(t_i)-\Delta(t_{i-1})\big)S(t_i)")
st.caption(
    "Black-Scholes assumes continuous, free hedging with a known volatility. "
    "This page breaks two of those assumptions: hedging only at discrete times, "
    "and hedging with the wrong volatility."
)

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Contract")
    S0 = st.slider("Spot S0", 50.0, 150.0, 100.0)
    K = st.slider("Strike K", 50.0, 150.0, 100.0)
    T = st.slider("Maturity T (years)", 0.1, 3.0, 1.0)
    r = st.slider("Rate r", 0.0, 0.10, 0.05, step=0.005)
    st.header("Volatility")
    sigma_hedge = st.slider("σ you sold at and hedge with", 0.05, 0.60, 0.20, step=0.01)
    sigma_true = st.slider("σ the stock really has", 0.05, 0.60, 0.20, step=0.01)
    st.header("Hedging")
    m = st.select_slider("Rebalances until maturity",
                         [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000], value=50)
    n_paths = st.select_slider("Simulated paths", [1_000, 5_000, 10_000, 25_000],
                               value=10_000)
    seed = st.number_input("Random seed", value=42, step=1)

if sigma_true != sigma_hedge:
    st.warning(f"Volatility mismatch: you hedge at {sigma_hedge:.0%} but the stock moves "
               f"at {sigma_true:.0%}. Expect a systematic profit or loss, not just noise.")


@st.cache_data(show_spinner="Running hedge simulation in C++...")
def pnl_sample(S0, K, T, r, s_true, s_hedge, m, n, seed):
    return delta_hedge_pnl(S0, K, T, r, s_true, s_hedge, m, n, seed)


tab_one, tab_dist, tab_freq, tab_vol = st.tabs(
    ["One hedge, step by step", "P&L distribution", "Rebalancing frequency", "Wrong volatility"]
)

# ---------------- Tab 1: walk through a single path (like book Figure 3.7) ----------------
with tab_one:
    path_seed = st.number_input("Path number", value=1, step=1, min_value=0)
    path = simulate_gbm(S0, r, sigma_true, T, m, 1, int(path_seed))[0]
    t = np.linspace(0.0, T, m + 1)

    # Same recursion as the C++ engine, written out in Python so we can record every step
    delta = np.empty(m + 1)
    value = np.empty(m + 1)
    pnl = np.empty(m + 1)
    delta[0] = bs_delta(OptionType.Call, S0, K, T, r, sigma_hedge)
    value[0] = bs_price(OptionType.Call, S0, K, T, r, sigma_hedge)
    pnl[0] = value[0] - delta[0] * S0
    for i in range(1, m):
        tau = T - t[i]
        delta[i] = bs_delta(OptionType.Call, path[i], K, tau, r, sigma_hedge)
        value[i] = bs_price(OptionType.Call, path[i], K, tau, r, sigma_hedge)
        pnl[i] = pnl[i - 1] * np.exp(r * (t[i] - t[i - 1])) - (delta[i] - delta[i - 1]) * path[i]
    payoff = max(path[-1] - K, 0.0)
    value[m] = payoff
    delta[m] = delta[m - 1]
    pnl[m] = pnl[m - 1] * np.exp(r * (t[m] - t[m - 1])) - payoff + delta[m - 1] * path[-1]

    status = "in the money" if path[-1] > K else "out of the money"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Premium received", f"{value[0]:.3f}")
    c2.metric("S(T)", f"{path[-1]:.2f}", help=f"The option ends {status}.")
    c3.metric("Payoff owed", f"{payoff:.3f}")
    c4.metric("Final P&L", f"{pnl[-1]:.4f}")

    fig = go.Figure()
    fig.add_scatter(x=t, y=path, name="Stock S(t)", line=dict(color=CALL_COLOR, width=2))
    fig.add_hline(y=K, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="strike K", annotation_position="bottom left")
    fig.update_layout(title="Stock path", xaxis_title="Time t", yaxis_title="Price",
                      height=320, margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=t, y=delta, name="Δ(t)", line=dict(color=ACCENT, width=2, shape="hv"))
    fig.update_layout(title="Shares held, Δ(t)", xaxis_title="Time t",
                      yaxis_title="Δ", yaxis_range=[-0.05, 1.05], height=320,
                      margin=dict(t=50), showlegend=False)
    left.plotly_chart(fig, width="stretch")

    # Hedge portfolio vs the option we owe: they should track each other.
    # At maturity, value the hedge BEFORE settling (cash grown one step + shares held),
    # so the gap between the two lines is exactly the final P&L.
    hedge_value = pnl + delta * path
    hedge_value[m] = pnl[m - 1] * np.exp(r * (t[m] - t[m - 1])) + delta[m - 1] * path[-1]
    fig = go.Figure()
    fig.add_scatter(x=t, y=value, name="Option we owe, V(t)",
                    line=dict(color=CALL_COLOR, width=2))
    fig.add_scatter(x=t, y=hedge_value, name="Our hedge: cash + Δ·S",
                    line=dict(color=ACCENT, width=2, dash="dash"))
    fig.update_layout(title="Does the hedge replicate the option?", xaxis_title="Time t",
                      yaxis_title="Value", height=320, margin=dict(t=50),
                      legend=dict(orientation="h", y=-0.3))
    right.plotly_chart(fig, width="stretch")
    st.caption(
        "Δ moves towards 1 when the stock goes up (the option will likely be exercised) "
        "and towards 0 when it goes down. The gap between the two lines on the right at "
        "maturity is the final P&L. Try fewer rebalances in the sidebar and watch the "
        "gap grow."
    )

# ---------------- Tab 2: distribution of final P&L ----------------
with tab_dist:
    pnl_all = pnl_sample(S0, K, T, r, sigma_true, sigma_hedge, m, n_paths, int(seed))
    premium = bs_price(OptionType.Call, S0, K, T, r, sigma_hedge)
    mean, sd = pnl_all.mean(), pnl_all.std(ddof=1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean P&L", f"{mean:.4f}", help="± is the 95% confidence interval")
    c1.caption(f"± {1.96 * sd / np.sqrt(n_paths):.4f}")
    c2.metric("Std of P&L", f"{sd:.4f}")
    c3.metric("Std as % of premium", f"{sd / premium:.1%}")

    fig = go.Figure()
    fig.add_histogram(x=pnl_all, nbinsx=80, name="Final P&L",
                      marker=dict(color=CALL_COLOR, opacity=0.75))
    fig.add_vline(x=0.0, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="break-even", annotation_position="top left")
    fig.add_vline(x=mean, line=dict(color=ACCENT, width=2), annotation_text="mean",
                  annotation_position="top right")
    fig.update_layout(xaxis_title="P&L at maturity", yaxis_title="Number of paths",
                      height=420, bargap=0.02, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{m} rebalances, {n_paths:,} paths. With the right volatility the mean is ~0 "
        "(book Example 3.3.1): hedging doesn't earn money, it removes risk. "
        "Banks make their profit by charging a spread on top of the premium."
    )

# ---------------- Tab 3: hedging error vs rebalancing frequency ----------------
with tab_freq:
    grid = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    stds = np.array([pnl_sample(S0, K, T, r, sigma_true, sigma_hedge, k, 5_000,
                                int(seed)).std(ddof=1) for k in grid])

    # Reference line with slope -1/2 through the m = 100 point
    ref = stds[grid.index(100)] * np.sqrt(100 / np.array(grid))

    fig = go.Figure()
    fig.add_scatter(x=grid, y=stds, name="Simulated std of P&L", mode="lines+markers",
                    line=dict(color=CALL_COLOR, width=2), marker=dict(size=9))
    fig.add_scatter(x=grid, y=ref, name="Theory: ∝ 1/√m", mode="lines",
                    line=dict(color=ACCENT, width=2, dash="dash"))
    fig.update_layout(xaxis_title="Rebalances m (log scale)",
                      yaxis_title="Std of final P&L (log scale)",
                      xaxis_type="log", yaxis_type="log", height=440)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "On log-log axes 1/√m is a straight line with slope −½: rebalancing 4× as often "
        "only halves the risk. In practice every rebalance costs a transaction fee, so "
        "desks trade off hedge accuracy against cost. That's the point of book "
        "Example 3.3.3. Each point uses 5,000 paths."
    )

# ---------------- Tab 4: hedging with the wrong volatility ----------------
with tab_vol:
    st.markdown(
        f"The stock really moves with **σ = {sigma_true:.0%}** (change it in the sidebar). "
        "What happens on average if you sold and hedged the option at a different volatility?"
    )
    vols = np.round(np.linspace(0.05, 0.60, 12), 3)
    means, half = [], []
    for v in vols:
        x = pnl_sample(S0, K, T, r, sigma_true, float(v), 250, 5_000, int(seed))
        means.append(x.mean())
        half.append(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
    means, half = np.array(means), np.array(half)
    theory = [(bs_price(OptionType.Call, S0, K, T, r, float(v))
               - bs_price(OptionType.Call, S0, K, T, r, sigma_true)) * np.exp(r * T)
              for v in vols]

    fig = go.Figure()
    fig.add_scatter(x=vols, y=means, name="Simulated mean P&L (95% CI)",
                    mode="markers", marker=dict(color=CALL_COLOR, size=10),
                    error_y=dict(type="data", array=half, color=CALL_COLOR))
    fig.add_scatter(x=vols, y=theory, name="(V(σ_hedge) − V(σ_true))·e^(rT)",
                    mode="lines", line=dict(color=ACCENT, width=2))
    fig.add_hline(y=0.0, line=dict(color=NEUTRAL, width=1, dash="dot"))
    fig.add_vline(x=sigma_true, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="true σ")
    fig.update_layout(xaxis_title="Volatility you sold at and hedged with",
                      yaxis_title="Mean P&L at maturity", xaxis_tickformat=".0%",
                      height=440)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Sell above the true volatility and you profit on average; sell below it and you "
        "lose. The average P&L is the mispricing itself, V(σ_hedge) − V(σ_true), grown at "
        "rate r to maturity. This is why options desks care so much about getting "
        "volatility right, and it sets up implied volatility in Step 2. "
        "Each point uses 250 rebalances and 5,000 paths."
    )
