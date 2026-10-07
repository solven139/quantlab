import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantlab import OptionType, bs_price, bs_vega, implied_vol, implied_vol_iterates

BLUE = "#3987e5"
ORANGE = "#d95926"
NEUTRAL = "#8a8985"

st.title("Implied Volatility")
st.markdown(
    "Markets quote option **prices**. The implied volatility is the σ that, "
    "plugged into Black-Scholes, reproduces that price (book Ch 4.1, eq. 4.1):"
)
st.latex(r"g(\sigma_{imp}) = V_{BS}(\sigma_{imp}) - V^{mkt} = 0")
st.caption(
    "No closed form exists, so it is found numerically: Newton-Raphson with vega as the "
    "derivative, falling back to bisection when Newton would leave the bracket "
    "known to contain the root."
)

with st.sidebar:
    st.header("Contract")
    cp_name = st.radio("Option", ["Call", "Put"], horizontal=True)
    cp = OptionType.Call if cp_name == "Call" else OptionType.Put
    S0 = st.number_input("Spot S0", value=100.0, min_value=1.0)
    K = st.number_input("Strike K", value=120.0, min_value=1.0)
    T = st.number_input("Maturity T (years)", value=1.0, min_value=0.01)
    r = st.number_input("Rate r", value=0.05, step=0.005, format="%.3f")

tab_solver, tab_fail, tab_smile = st.tabs(
    ["The solver, step by step", "Why plain Newton isn't enough", "The volatility smile"]
)

# ---------------- Tab 1: watch the combined solver converge ----------------
with tab_solver:
    df = np.exp(-r * T)
    lower = max(S0 - K * df, 0.0) if cp == OptionType.Call else max(K * df - S0, 0.0)
    upper = S0 if cp == OptionType.Call else K * df
    st.markdown(f"Valid prices for this {cp_name.lower()}: strictly between "
                f"**{lower:.4f}** (intrinsic value) and **{upper:.4f}**.")
    price = st.number_input("Market price", value=2.0, min_value=0.0, step=0.1,
                            format="%.4f",
                            help="Book Example 4.1.2: a call with S0=100, K=120, T=1, "
                                 "r=5% trading at 2 has σ_imp = 0.161482728841394")
    try:
        steps = implied_vol_iterates(cp, price, S0, K, T, r)
    except ValueError as e:
        st.error(f"No implied volatility exists: {e}. A price outside the no-arbitrage "
                 "bounds would let someone make riskless profit.")
        st.stop()

    iv = steps[-1]
    # For comparison: how many steps would pure bisection on [0.0001, 5] need?
    lo, hi, n_bis = 1e-4, 5.0, 0
    while True:
        mid = 0.5 * (lo + hi)
        n_bis += 1
        f = bs_price(cp, S0, K, T, r, mid) - price
        if abs(f) < 1e-12 or hi - lo < 1e-15 or n_bis > 200:
            break
        lo, hi = (lo, mid) if f > 0 else (mid, hi)

    c1, c2, c3 = st.columns(3)
    c1.metric("Implied volatility", f"{iv:.10f}")
    c2.metric("Combined method: steps", len(steps) - 1)
    c3.metric("Pure bisection: steps", n_bis)

    # g(sigma) curve with the iterates marked on it
    # Zoom in on the region the solver actually visited
    lo_x = max(0.005, min(steps) - 0.1)
    hi_x = max(steps) + 0.1
    grid = np.linspace(lo_x, hi_x, 400)
    g = [bs_price(cp, S0, K, T, r, s) - price for s in grid]
    fig = go.Figure()
    fig.add_scatter(x=grid, y=g, name="g(σ) = V_BS(σ) − market price",
                    line=dict(color=BLUE, width=2))
    fig.add_hline(y=0.0, line=dict(color=NEUTRAL, width=1, dash="dot"))
    g_steps = [bs_price(cp, S0, K, T, r, s) - price for s in steps]
    fig.add_scatter(x=steps, y=g_steps, mode="markers+text", name="Solver iterates",
                    text=[str(i) for i in range(len(steps))], textposition="top center",
                    marker=dict(color=ORANGE, size=10))
    # Newton tangent lines: from (σ_k, g_k) down to where they cross zero
    for k in range(len(steps) - 1):
        v = bs_vega(S0, K, T, r, steps[k])
        x_cross = steps[k] - g_steps[k] / v
        fig.add_scatter(x=[steps[k], x_cross], y=[g_steps[k], 0.0], mode="lines",
                        line=dict(color=ORANGE, width=1, dash="dash"),
                        showlegend=(k == 0), name="Newton tangent", hoverinfo="skip")
    fig.update_layout(xaxis_title="σ", yaxis_title="g(σ)", xaxis_tickformat=".0%",
                      xaxis_range=[lo_x, hi_x], height=430)
    st.plotly_chart(fig, width="stretch")

    errors = [abs(s - iv) for s in steps]
    st.dataframe(
        {"step": list(range(len(steps))),
         "σ": [f"{s:.15f}" for s in steps],
         "|σ − σ_imp|": [f"{e:.2e}" for e in errors]},
        hide_index=True, width="stretch",
    )
    st.caption(
        "Watch the error column: near the root the number of correct digits roughly "
        "doubles every Newton step (quadratic convergence). Bisection only halves the "
        "error each step (linear convergence)."
    )

# ---------------- Tab 2: where plain Newton breaks down ----------------
with tab_fail:
    st.markdown(
        "Newton divides by vega. Far from the money vega is tiny, so the first step can "
        "fling σ far away. Here a deep out-of-the-money call (S0=100, K=250, T=0.5, r=0) "
        "with true σ = 60%, starting from the same guess of 20%:"
    )
    S0b, Kb, Tb, rb, true_sigma = 100.0, 250.0, 0.5, 0.0, 0.6
    pb = bs_price(OptionType.Call, S0b, Kb, Tb, rb, true_sigma)

    plain = [0.2]
    status = "converged"
    for _ in range(30):
        s = plain[-1]
        f = bs_price(OptionType.Call, S0b, Kb, Tb, rb, s) - pb
        if abs(f) < 1e-12:
            break
        nxt = s - f / bs_vega(S0b, Kb, Tb, rb, s)
        plain.append(nxt)
        if not (0.0 < nxt < 10.0):
            status = "diverged"
            break
    combined = list(implied_vol_iterates(OptionType.Call, pb, S0b, Kb, Tb, rb))

    c1, c2 = st.columns(2)
    c1.metric(f"Plain Newton ({len(plain) - 1} step)", status)
    c2.metric(f"Combined method ({len(combined) - 1} steps)", "converged")

    fig = go.Figure()
    fig.add_scatter(x=list(range(len(plain))), y=plain, name="Plain Newton",
                    mode="lines+markers", line=dict(color=ORANGE, width=2),
                    marker=dict(size=9))
    fig.add_scatter(x=list(range(len(combined))), y=combined, name="Combined (your C++)",
                    mode="lines+markers", line=dict(color=BLUE, width=2),
                    marker=dict(size=9))
    fig.add_hline(y=true_sigma, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="true σ = 60%")
    fig.update_layout(xaxis_title="Step", yaxis_title="σ (log scale)", yaxis_type="log",
                      height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Vega at the starting guess is only {bs_vega(S0b, Kb, Tb, rb, 0.2):.2e}, so the "
        f"first Newton step jumps to σ ≈ {plain[1]:,.0f}, "
        "a volatility of over a billion percent (note the log scale). The combined method rejects that "
        "step because it leaves the bracket, bisects instead, and then lets Newton finish "
        "once it is close."
    )

# ---------------- Tab 3: the smile (preview of why Chapters 4-10 exist) ----------------
with tab_smile:
    st.markdown(
        "If the market really followed Black-Scholes, every strike would give the **same** "
        "implied vol: a flat line. Real markets show a **smile** or **skew**.\n\n"
        "A simple way to break the constant-σ assumption: suppose volatility is "
        "**uncertain**. With probability p the stock moves at a low volatility, otherwise "
        "at a high one. The fair price is then the mixture of two Black-Scholes prices. "
        "Invert those prices strike by strike and see what comes out."
    )
    c1, c2, c3 = st.columns(3)
    s_low = c1.slider("Low volatility", 0.05, 0.50, 0.15, step=0.01)
    s_high = c2.slider("High volatility", 0.05, 0.80, 0.40, step=0.01)
    p = c3.slider("Probability of low volatility", 0.0, 1.0, 0.7, step=0.05)

    strikes = np.linspace(0.6 * S0, 1.6 * S0, 41)
    ivs, used = [], []
    for k in strikes:
        # Use the out-of-the-money option at each strike: it carries the most time value
        typ = OptionType.Put if k < S0 else OptionType.Call
        mix = (p * bs_price(typ, S0, k, T, r, s_low)
               + (1 - p) * bs_price(typ, S0, k, T, r, s_high))
        try:
            ivs.append(implied_vol(typ, mix, S0, k, T, r))
            used.append(k)
        except ValueError:
            pass
    avg_var = np.sqrt(p * s_low**2 + (1 - p) * s_high**2)

    fig = go.Figure()
    fig.add_scatter(x=used, y=ivs, name="Implied vol of the mixture prices",
                    mode="lines+markers", line=dict(color=BLUE, width=2),
                    marker=dict(size=8))
    fig.add_hline(y=avg_var, line=dict(color=ORANGE, width=2, dash="dash"),
                  annotation_text="√(average variance)", annotation_position="top left")
    fwd = S0 * np.exp(r * T)
    fig.add_vline(x=fwd, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="forward S0·e^(rT)")
    fig.update_layout(xaxis_title="Strike K", yaxis_title="Implied volatility",
                      yaxis_tickformat=".1%", height=430)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Uncertain volatility fattens both tails of the return distribution, so options "
        "far from the money are worth more than one constant σ can explain: a smile. "
        "Its bottom sits at the forward price S0·e^(rT), not at S0. "
        "Set both volatilities equal and the smile goes flat. That's Black-Scholes again. "
        "Real equity markets show a skew (higher implied vol for low strikes); jumps "
        "(Step 4) and Heston's correlated stochastic volatility (Step 5) produce it."
    )

    # ---------------- 3D: the implied volatility surface ----------------
    st.subheader("The volatility surface: a smile for every maturity")
    st.markdown(
        "Repeat the same inversion for many maturities and the smiles stack up into a "
        "surface (book Figure 4.3). Drag to rotate, scroll to zoom, hover for values."
    )
    c1, c2 = st.columns(2)
    axis_mode = c1.radio("Horizontal axis", ["Strike K", "Moneyness K / forward"],
                         horizontal=True)
    show_flat = c2.checkbox("Show the flat Black-Scholes plane for comparison", value=True)

    mats = np.linspace(0.1, 3.0, 24)
    xs = np.linspace(0.6, 1.6, 31)           # strikes (as a multiple of S0) or moneyness
    Z = np.full((len(mats), len(xs)), np.nan)
    X = np.empty_like(Z)
    for i, tau in enumerate(mats):
        F = S0 * np.exp(r * tau)
        for j, x_ in enumerate(xs):
            # Same strikes for every maturity, or same K/F for every maturity
            k = x_ * S0 if axis_mode == "Strike K" else x_ * F
            X[i, j] = k if axis_mode == "Strike K" else x_
            typ = OptionType.Put if k < F else OptionType.Call   # out-of-the-money side
            mix = (p * bs_price(typ, S0, k, tau, r, s_low)
                   + (1 - p) * bs_price(typ, S0, k, tau, r, s_high))
            try:
                Z[i, j] = implied_vol(typ, mix, S0, k, tau, r)
            except ValueError:
                pass  # no time value left: leave a hole rather than invent a number
    Y = np.repeat(mats[:, None], len(xs), axis=1)

    fig = go.Figure()
    fig.add_surface(
        x=X, y=Y, z=Z, colorscale="Blues", reversescale=False,
        colorbar=dict(title="Implied vol", tickformat=".0%", len=0.7),
        hovertemplate=(f"{'K' if axis_mode == 'Strike K' else 'K/F'}: %{{x:.2f}}<br>"
                       "T: %{y:.2f} y<br>Implied vol: %{z:.2%}"
                       "<extra></extra>"),
        name="Implied vol",
    )
    if show_flat:
        fig.add_surface(
            x=X, y=Y, z=np.full_like(Z, avg_var), showscale=False, opacity=0.35,
            colorscale=[[0, ORANGE], [1, ORANGE]], name="Flat σ",
            hovertemplate="Black-Scholes: one σ = %{z:.2%}<extra></extra>",
        )
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis_title=axis_mode, yaxis_title="Maturity T (years)",
            zaxis_title="Implied vol", zaxis_tickformat=".0%",
            camera=dict(eye=dict(x=0.4, y=-2.1, z=0.8)),
        ),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Read it two ways. Fix a maturity (a slice along the strike axis): that's the "
        "smile from the chart above. Fix a strike and move along maturity: the "
        "term structure. Here the smile is strongest for short maturities and flattens "
        "for long ones. Under Black-Scholes the whole surface would be the flat orange "
        "plane."
    )