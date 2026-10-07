import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantlab import (HestonModel, MertonModel, OptionType, cos_price, implied_vol,
                      simulate_heston)

BLUE = "#3987e5"
ORANGE = "#d95926"
NEUTRAL = "#8a8985"
GREEN = "#1baf7a"
VIOLET = "#4a3aa7"
PATH_BLUE = "rgba(57, 135, 229, 0.35)"
PATH_ORANGE = "rgba(217, 89, 38, 0.35)"

st.title("Heston Stochastic Volatility")
st.markdown(
    "In Black-Scholes, volatility is a constant. In Heston it is random: the **variance** "
    "v(t) follows its own mean-reverting process, and it is correlated with the stock "
    "(book Ch 8)."
)
st.latex(r"\frac{dS}{S} = r\,dt + \sqrt{v}\,dW_x,\qquad "
         r"dv = \kappa(\bar v - v)\,dt + \gamma\sqrt{v}\,dW_v,\qquad dW_x\,dW_v = \rho\,dt")

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Heston parameters")
    preset = st.selectbox("Start from", ["Typical equity", "Fang & Oosterlee (book test case)"])
    if preset.startswith("Typical"):
        d = dict(kappa=1.5, vbar=0.04, gamma=0.6, rho=-0.7, v0=0.04)
    else:
        d = dict(kappa=1.5768, vbar=0.0398, gamma=0.5751, rho=-0.5711, v0=0.0175)
    kappa = st.slider("κ  mean-reversion speed", 0.1, 5.0, d["kappa"], step=0.01)
    vbar = st.slider("v̄  long-run variance", 0.005, 0.20, d["vbar"], step=0.001, format="%.3f")
    gamma = st.slider("γ  volatility of variance", 0.01, 1.5, d["gamma"], step=0.01)
    rho = st.slider("ρ  stock–variance correlation", -0.95, 0.95, d["rho"], step=0.01)
    v0 = st.slider("v₀  today's variance", 0.005, 0.20, d["v0"], step=0.001, format="%.3f")
    st.header("Market")
    S0 = st.number_input("Spot S0", value=100.0, min_value=1.0)
    r = st.slider("Rate r", 0.0, 0.10, 0.03, step=0.005)

model = HestonModel(r=r, kappa=kappa, vbar=vbar, gamma=gamma, rho=rho, v0=v0)

c1, c2, c3 = st.columns(3)
c1.metric("Today's volatility √v₀", f"{math.sqrt(v0):.1%}")
c2.metric("Long-run volatility √v̄", f"{math.sqrt(vbar):.1%}")
feller = 2 * kappa * vbar
if model.feller_satisfied():
    c3.metric("Feller: 2κv̄ ≥ γ²", "satisfied")
    c3.caption(f"2κv̄ = {feller:.3f} ≥ γ² = {gamma**2:.3f}: v(t) stays above 0")
else:
    c3.metric("Feller: 2κv̄ ≥ γ²", "violated")
    c3.caption(f"2κv̄ = {feller:.3f} < γ² = {gamma**2:.3f}: v(t) can hit 0 "
               "(common in real market fits)")


def iv_at(m, mny, T):
    """Implied vol at moneyness K/F from COS prices of the out-of-the-money option."""
    F = S0 * math.exp(r * T)
    K = mny * F
    typ = OptionType.Put if K < F else OptionType.Call
    price = cos_price(m, typ, S0, K, T, N=256, L=12.0)
    return implied_vol(typ, price, S0, K, T, r)


tab_paths, tab_smile, tab_vs, tab_surf, tab_check = st.tabs(
    ["Paths: stock and variance", "How each parameter shapes the smile",
     "Heston vs jumps", "Volatility surface", "Pricing checks"]
)

# ---------------- Tab 1: two coupled processes ----------------
with tab_paths:
    a, b = st.columns(2)
    T_p = a.slider("Horizon (years)", 0.5, 5.0, 2.0, step=0.5)
    n_draw = b.slider("Paths drawn", 5, 60, 20)
    n_steps = 500
    S, v, zero_hits = simulate_heston(S0, r, kappa, vbar, gamma, rho, v0, T_p, n_steps,
                                      n_draw, 3)
    t = np.linspace(0.0, T_p, n_steps + 1)
    hl = 1  # highlight one path in both charts so the link between them is visible

    fig = go.Figure()
    for j, row in enumerate(S):
        fig.add_scatter(x=t, y=row, mode="lines", showlegend=(j == hl),
                        name="one path, highlighted in both charts" if j == hl else None,
                        line=dict(color=BLUE if j == hl else PATH_BLUE, width=2.5 if j == hl else 1),
                        hoverinfo="skip")
    fig.update_layout(title="Stock S(t)", xaxis_title="Time t (years)", yaxis_title="S(t)",
                      height=330, margin=dict(t=50), legend=dict(y=1.15, orientation="h"))
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for j, row in enumerate(v):
        fig.add_scatter(x=t, y=np.sqrt(row), mode="lines", showlegend=False,
                        line=dict(color=ORANGE if j == hl else PATH_ORANGE,
                                  width=2.5 if j == hl else 1), hoverinfo="skip")
    fig.add_hline(y=math.sqrt(vbar), line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="long-run √v̄", annotation_position="top left")
    fig.update_layout(title="Volatility √v(t)", xaxis_title="Time t (years)",
                      yaxis_title="√v(t)", yaxis_tickformat=".0%", height=330,
                      margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")

    if zero_hits > 0:
        st.warning(f"The variance tried to go below zero {zero_hits:,} times in these paths "
                   "(Euler scheme, floored at 0). With the Feller condition violated, v(t) "
                   "really does touch 0, and simple schemes struggle there. Step 6 compares "
                   "better simulation schemes.")
    st.caption(
        "Follow the highlighted path. With ρ < 0, when the stock falls the volatility tends "
        "to jump up, and vice versa: the 'leverage effect' seen in equity markets. Volatility "
        "is pulled back towards √v̄ at speed κ, and γ controls how wildly it moves."
    )

# ---------------- Tab 2: what each parameter does to the smile ----------------
with tab_smile:
    T_s = st.slider("Maturity (years)", 0.1, 3.0, 0.5, step=0.1, key="T_smile")
    param = st.radio("Vary", ["ρ (correlation)", "γ (vol of vol)", "v₀ (today's variance)",
                              "κ (mean reversion)"], horizontal=True)
    mny = np.linspace(0.7, 1.3, 31)
    base = dict(kappa=kappa, vbar=vbar, gamma=gamma, rho=rho, v0=v0)
    if param.startswith("ρ"):
        key, values = "rho", [-0.9, -0.5, 0.0, 0.5]
    elif param.startswith("γ"):
        key, values = "gamma", [0.1, 0.4, 0.8, 1.2]
    elif param.startswith("v₀"):
        key, values = "v0", [0.01, 0.04, 0.09]
    else:
        key, values = "kappa", [0.3, 1.5, 5.0]

    fig = go.Figure()
    for colour, val in zip([BLUE, ORANGE, GREEN, VIOLET], values):
        p_ = dict(base, **{key: val})
        m_ = HestonModel(r=r, **p_)
        ivs = []
        for x in mny:
            try:
                ivs.append(iv_at(m_, x, T_s))
            except ValueError:
                ivs.append(None)
        fig.add_scatter(x=mny, y=ivs, name=f"{key} = {val:g}", mode="lines+markers",
                        line=dict(color=colour, width=2), marker=dict(size=5))
    fig.update_layout(xaxis_title="Moneyness K / forward", yaxis_title="Implied volatility",
                      yaxis_tickformat=".1%", height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption({
        "rho": "ρ tilts the smile. Negative ρ: when the stock falls, volatility rises, so low "
               "strikes (crash protection) get expensive: a downward skew. ρ = 0 gives a "
               "symmetric smile; positive ρ tilts it the other way.",
        "gamma": "γ bends the smile. More vol of vol means fatter tails on both sides, so the "
                 "wings rise. γ → 0 flattens it towards Black-Scholes.",
        "v0": "v₀ moves the short end up and down: today's volatility level. For long "
              "maturities the level is set more by v̄.",
        "kappa": "κ controls how quickly shocks to volatility die out. Fast mean reversion "
                 "averages the randomness away, so the smile flattens.",
    }[key])

# ---------------- Tab 3: skew term structure, Heston vs Merton ----------------
with tab_vs:
    st.markdown(
        "Both a jump model and Heston can produce a downward skew. The difference is **how the "
        "skew changes with maturity**. Here the skew is measured as the slope of implied vol "
        "around the money, (σ(0.95F) − σ(1.05F)) / log(1.05/0.95), and each curve is "
        "scaled to 100% at 3 months so only the shape is compared."
    )
    jm = MertonModel(r=r, sigma=0.15, xi=0.5, mu_j=-0.1, sigma_j=0.15)
    mats = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    dk = math.log(1.05 / 0.95)

    def skew(m_, T):
        return (iv_at(m_, 0.95, T) - iv_at(m_, 1.05, T)) / dk

    sh = np.array([skew(model, T) for T in mats])
    sm = np.array([skew(jm, T) for T in mats])
    i_ref = int(np.where(mats == 0.25)[0][0])

    fig = go.Figure()
    fig.add_scatter(x=mats, y=sh / sh[i_ref], name="Heston (your sidebar parameters)",
                    mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8))
    fig.add_scatter(x=mats, y=sm / sm[i_ref], name="Merton jumps (Step 4 defaults)",
                    mode="lines+markers", line=dict(color=ORANGE, width=2), marker=dict(size=8))
    fig.add_vline(x=0.25, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="3 months = 100%")
    fig.update_layout(xaxis_title="Maturity T (years)", yaxis_title="Skew relative to 3 months",
                      yaxis_tickformat=".0%", height=440)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"At 5 years the Heston skew is still {sh[-1] / sh[i_ref]:.0%} of its 3-month value; "
        f"the Merton skew has fallen to {sm[-1] / sm[i_ref]:.0%}. Jumps average out over long "
        "horizons (central limit theorem), while correlated stochastic volatility keeps "
        "skewing returns. At the very short end it's the reverse: jump skews keep steepening "
        "as T → 0, Heston's levels off. Real markets need both, which is why practitioners "
        "combine them (the Bates model = Heston + Merton jumps)."
    )

# ---------------- Tab 4: 3D surface ----------------
with tab_surf:
    st.markdown("Every point: a COS price under Heston, inverted with your implied-vol solver.")
    mats3 = np.linspace(0.1, 3.0, 20)
    grid = np.linspace(0.7, 1.3, 25)
    Z = np.full((len(mats3), len(grid)), np.nan)
    for i_, T in enumerate(mats3):
        for j_, x in enumerate(grid):
            try:
                Z[i_, j_] = iv_at(model, x, T)
            except ValueError:
                pass
    fig = go.Figure(go.Surface(
        x=np.tile(grid, (len(mats3), 1)), y=np.repeat(mats3[:, None], len(grid), axis=1), z=Z,
        colorscale="Blues", colorbar=dict(title="Implied vol", tickformat=".0%", len=0.7),
        hovertemplate="K/F: %{x:.2f}<br>T: %{y:.2f} y<br>Implied vol: %{z:.2%}<extra></extra>"))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(xaxis_title="Moneyness K / forward",
                                 yaxis_title="Maturity T (years)", zaxis_title="Implied vol",
                                 zaxis_tickformat=".0%",
                                 camera=dict(eye=dict(x=0.4, y=-2.1, z=0.8))))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Compare with the Jump Models surface: here the skew stays visible across the whole "
        "maturity range. Set v₀ well below v̄ and the at-the-money level rises with maturity "
        "(an upward term structure), because variance is expected to revert up to v̄."
    )

# ---------------- Tab 5: checks ----------------
with tab_check:
    st.markdown(
        "**1. Book / literature reference.** With the Fang & Oosterlee parameters "
        "(κ=1.5768, v̄=0.0398, γ=0.5751, ρ=−0.5711, v₀=0.0175, r=0, S0=K=100, T=1), the "
        "reference call price is **5.785155450**."
    )
    fo = HestonModel(r=0.0, kappa=1.5768, vbar=0.0398, gamma=0.5751, rho=-0.5711, v0=0.0175)
    rows = []
    for N in [32, 64, 128, 256]:
        val = cos_price(fo, OptionType.Call, 100.0, 100.0, 1.0, N=N, L=12.0)
        rows.append({"COS terms N": N, "COS price": f"{val:.9f}",
                     "Error vs reference": f"{abs(val - 5.785155450):.1e}"})
    st.dataframe(rows, hide_index=True, width="stretch")

    st.markdown("**2. Monte Carlo with your sidebar parameters** (Euler, 250 steps, 100,000 paths)")
    T_c = st.slider("Maturity (years)", 0.25, 3.0, 1.0, step=0.25, key="T_check")
    S_mc, _, _ = simulate_heston(S0, r, kappa, vbar, gamma, rho, v0, T_c, 250, 100_000, 11)
    ST = S_mc[:, -1]
    rows = []
    n_biased = 0
    for K in [0.8 * S0, S0, 1.2 * S0]:
        pay = math.exp(-r * T_c) * np.maximum(ST - K, 0.0)
        se = pay.std(ddof=1) / math.sqrt(len(pay))
        v_cos = cos_price(model, OptionType.Call, S0, K, T_c, N=256, L=12.0)
        outside = abs(pay.mean() - v_cos) > 1.96 * se
        n_biased += outside
        rows.append({"Strike": f"{K:g}", "COS": f"{v_cos:.4f}",
                     "Monte Carlo": f"{pay.mean():.4f} ± {1.96 * se:.4f}",
                     "Difference": f"{pay.mean() - v_cos:+.4f}",
                     "Within 95% CI?": "no" if outside else "yes"})
    st.dataframe(rows, hide_index=True, width="stretch")
    if n_biased:
        st.warning(f"{n_biased} of 3 differences are larger than the Monte Carlo noise. That "
                   "isn't bad luck: it is discretisation bias of the Euler scheme, which gets "
                   "worse when the Feller condition is violated (v keeps hitting 0 and is "
                   "floored). More paths won't fix it; a better scheme will (Step 6).")
    st.caption(
        "COS is essentially exact; the Monte Carlo column has statistical noise (the ±) and a "
        "small bias from the Euler time-stepping, which grows when the Feller condition is "
        "violated. Measuring and removing that bias is Step 6."
    )
