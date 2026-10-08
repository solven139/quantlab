import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantlab import (CirScheme, HestonModel, LocalVolSurface, OptionType, cos_price,
                      heston_at_times, implied_vol, local_vol_at_times)

BLUE = "#3987e5"
ORANGE = "#d95926"
GREEN = "#1baf7a"
VIOLET = "#4a3aa7"
NEUTRAL = "#8a8985"

st.title("Local Volatility")
st.markdown(
    "Heston explains the smile with **random** volatility. Local volatility uses a "
    "**deterministic** function instead: volatility depends only on where the stock is and "
    "when, σ_LV(S, t). Dupire showed that today's option prices pin this function down "
    "exactly (book eq. 4.49):"
)
st.latex(r"\sigma_{LV}^2(T,K) = \frac{\partial C/\partial T + rK\,\partial C/\partial K}"
         r"{\tfrac12 K^2\,\partial^2 C/\partial K^2},\qquad "
         r"\frac{dS}{S} = r\,dt + \sigma_{LV}(S,t)\,dW")
st.caption(
    "The experiment of book Ch 10.2: Heston generates the 'market' prices, Dupire turns them "
    "into a local-vol surface, and then both models are compared, first on today's options, "
    "then on options that start in the future."
)

# ---------------- Inputs: the Heston "market" ----------------
with st.sidebar:
    st.header("Heston 'market'")
    st.caption("Defaults: book Ch 10.2 (Fig 10.2).")
    kappa = st.slider("κ", 0.1, 5.0, 1.3, step=0.1)
    vbar = st.slider("v̄", 0.01, 0.20, 0.05, step=0.005, format="%.3f")
    gamma = st.slider("γ", 0.05, 1.0, 0.3, step=0.05)
    rho = st.slider("ρ", -0.95, 0.5, -0.3, step=0.05)
    v0 = st.slider("v₀", 0.01, 0.20, 0.10, step=0.005, format="%.3f")
    S0 = 1.0
    r = 0.0
    st.caption("S0 = 1 and r = 0, as in the book.")

heston = HestonModel(r=r, kappa=kappa, vbar=vbar, gamma=gamma, rho=rho, v0=v0)
KS = list(np.linspace(0.3, 2.5, 45) * S0)
TS = list(np.linspace(0.05, 3.0, 30))


@st.cache_resource(show_spinner="Building the local volatility surface (Dupire, in C++)...")
def build_surface(kappa, vbar, gamma, rho, v0):
    m = HestonModel(r=r, kappa=kappa, vbar=vbar, gamma=gamma, rho=rho, v0=v0)
    return LocalVolSurface(m, S0, KS, TS)


surface = build_surface(kappa, vbar, gamma, rho, v0)


def iv_from_price(price, k, T, typ):
    try:
        return implied_vol(typ, price, S0, k, T, r)
    except ValueError:
        return None


def heston_iv(k, T):
    typ = OptionType.Put if k < S0 else OptionType.Call
    return iv_from_price(cos_price(heston, typ, S0, k, T, N=256, L=12.0), k, T, typ)


def mc_iv(ST, k, T, disc=1.0):
    typ = OptionType.Put if k < S0 else OptionType.Call
    pay = np.maximum(ST - k, 0.0) if typ == OptionType.Call else np.maximum(k - ST, 0.0)
    return iv_from_price(disc * pay.mean(), k, T, typ)


tab_surf, tab_today, tab_future = st.tabs(
    ["The local vol surface", "Same prices today", "Different future"]
)

# ---------------- Tab 1: the Dupire surface ----------------
with tab_surf:
    V = surface.values()
    mask = (np.array(KS) >= 0.5 * S0) & (np.array(KS) <= 1.8 * S0)
    Kg = np.array(KS)[mask]
    fig = go.Figure(go.Surface(
        x=np.tile(Kg, (len(TS), 1)), y=np.repeat(np.array(TS)[:, None], len(Kg), axis=1),
        z=V[:, mask], colorscale="Blues",
        colorbar=dict(title="σ_LV", tickformat=".0%", len=0.7),
        hovertemplate="S: %{x:.2f}<br>t: %{y:.2f} y<br>σ_LV: %{z:.2%}<extra></extra>"))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(xaxis_title="Stock level S", yaxis_title="Time t (years)",
                                 zaxis_title="Local vol", zaxis_tickformat=".0%",
                                 camera=dict(eye=dict(x=0.4, y=-2.1, z=0.8))))
    st.plotly_chart(fig, width="stretch")

    T_slice = st.select_slider("Compare at maturity T", [0.25, 0.5, 1.0, 2.0], value=1.0)
    ks = np.linspace(0.6, 1.5, 37) * S0
    fig = go.Figure()
    fig.add_scatter(x=ks, y=[heston_iv(k, T_slice) for k in ks], name="Implied vol σ_imp(K, T)",
                    line=dict(color=ORANGE, width=2))
    fig.add_scatter(x=ks, y=[surface.sigma(k, T_slice) for k in ks],
                    name="Local vol σ_LV(S = K, t = T)", line=dict(color=BLUE, width=2))
    fig.update_layout(xaxis_title="Strike K / stock level S", yaxis_title="Volatility",
                      yaxis_tickformat=".0%", height=400)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Local vol is steeper than implied vol: roughly twice the slope near the money. Implied "
        "vol is an average of local vols along the paths to the strike, so the local surface has "
        "to tilt harder to produce the same smile. The far wings (where the density ∂²C/∂K² is "
        "almost 0) are filled from the nearest reliable value; book Ch 4.3 warns that Dupire is "
        "unreliable there."
    )

# ---------------- Tab 2: same vanilla prices ----------------
with tab_today:
    st.markdown("Simulate the local-vol model and price European options at maturity T. "
                "If Dupire worked, its smile matches Heston's.")
    c1, c2 = st.columns(2)
    T_today = c1.select_slider("Maturity T", [0.25, 0.5, 1.0, 2.0], value=1.0, key="T_today")
    n_today = c2.select_slider("Local-vol paths", [20_000, 50_000, 100_000], value=50_000)

    @st.cache_data(show_spinner="Simulating local vol paths...")
    def lv_terminal(params, T, n):
        return local_vol_at_times(surface, S0, r, [T], 100, n, 1)[:, 0]

    ST = lv_terminal((kappa, vbar, gamma, rho, v0), T_today, n_today)
    ks = np.linspace(0.7, 1.4, 15) * S0
    ivh = [heston_iv(k, T_today) for k in ks]
    ivl = [mc_iv(ST, k, T_today) for k in ks]
    fig = go.Figure()
    fig.add_scatter(x=ks, y=ivh, name="Heston (COS, exact)", line=dict(color=ORANGE, width=2))
    fig.add_scatter(x=ks, y=ivl, name="Local vol (Monte Carlo)", mode="markers",
                    marker=dict(color=BLUE, size=9))
    fig.update_layout(xaxis_title="Strike K", yaxis_title="Implied volatility",
                      yaxis_tickformat=".1%", height=420)
    st.plotly_chart(fig, width="stretch")
    diffs = [abs(a - b) for a, b in zip(ivh, ivl) if a is not None and b is not None]
    st.success(f"Largest gap between the two smiles: {max(diffs) * 100:.2f} vol points. "
               "Same vanilla prices: as far as today's market is concerned, the two models are "
               "interchangeable. (Book Fig 10.2a.)")

# ---------------- Tab 3: forward-start options ----------------
with tab_future:
    st.markdown(
        "A **forward-start** option pays (S(T₂)/S(T₁) − k)⁺: its strike is fixed only at T₁, "
        "as a percentage of the stock price then. Its value depends on the smile **as seen at "
        "T₁**, which today's vanilla prices don't determine. Here the two models part ways "
        "(book Ch 10.1–10.2)."
    )
    c1, c2, c3 = st.columns(3)
    T1 = c1.select_slider("Start T₁", [0.5, 1.0, 1.5, 2.0], value=1.0)
    tau = c2.select_slider("Length T₂ − T₁", [0.5, 1.0], value=1.0)
    n_fwd = c3.select_slider("Paths per model", [20_000, 50_000, 100_000], value=50_000)
    T2 = T1 + tau

    @st.cache_data(show_spinner="Simulating Heston (QE) and local vol to T₁ and T₂...")
    def two_dates(params, T1, T2, n):
        k_, vb_, g_, rh_, v0_ = params
        ph = heston_at_times(CirScheme.QE, S0, r, k_, vb_, g_, rh_, v0_, [T1, T2], 64, n, 2)
        pl = local_vol_at_times(surface, S0, r, [T1, T2], 64, n, 3)
        return ph, pl

    ph, pl = two_dates((kappa, vbar, gamma, rho, v0), T1, T2, n_fwd)
    kf = np.linspace(0.7, 1.35, 14)

    def fwd_smile(P):
        R = P[:, 1] / P[:, 0]
        return [mc_iv(R, k, tau) for k in kf]   # S0 = 1 and r = 0: value of (R - k)+ is BS(1, k, tau)

    fh, fl = fwd_smile(ph), fwd_smile(pl)
    today = [heston_iv(k * S0, tau) for k in kf]

    fig = go.Figure()
    fig.add_scatter(x=kf, y=fh, name=f"Heston forward smile ({T1:g}y → {T2:g}y)",
                    mode="lines+markers", line=dict(color=ORANGE, width=2), marker=dict(size=7))
    fig.add_scatter(x=kf, y=fl, name=f"Local vol forward smile ({T1:g}y → {T2:g}y)",
                    mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=7))
    fig.add_scatter(x=kf, y=today, name=f"Today's {tau:g}y smile (both models)",
                    line=dict(color=NEUTRAL, width=1.5, dash="dot"))
    fig.update_layout(xaxis_title="Forward strike k (as a fraction of S(T₁))",
                      yaxis_title="Forward implied volatility", yaxis_tickformat=".1%",
                      height=460)
    st.plotly_chart(fig, width="stretch")

    def span(x):
        x = [v for v in x if v is not None]
        return (max(x) - min(x)) * 100

    c1, c2 = st.columns(2)
    c1.metric("Heston forward smile: range", f"{span(fh):.1f} vol pts")
    c2.metric("Local vol forward smile: range", f"{span(fl):.1f} vol pts")
    st.caption(
        "In Heston the smile is generated by random volatility, so it regenerates: a year from "
        "now there is still a smile of similar shape (book Fig 10.1). In local vol the smile "
        "lives on the fixed surface σ_LV(S, t), and that surface flattens as t grows, so the "
        "forward smile comes out flatter (book Fig 10.2b). Same prices today, different "
        "future: that's why exotics depending on future smiles (forward starts, cliquets, "
        "barriers) are priced with stochastic or stochastic-local volatility models (Ch 10.2). "
        f"(Both forward smiles sit {'below' if v0 > vbar else 'above'} today's smile because "
        f"v₀ = {v0:.3f} is "
        f"{'above' if v0 > vbar else 'below'} v̄ = {vbar:.3f}: variance is expected to "
        f"{'fall' if v0 > vbar else 'rise'} by T₁. Set v₀ = v̄ to remove that level effect.)"
    )