import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantlab import (CirScheme, HestonModel, OptionType, bs_price, cir_step_samples, cos_price,
                      gbm_schemes, heston_terminal)

BLUE = "#3987e5"
ORANGE = "#d95926"
GREEN = "#1baf7a"
NEUTRAL = "#8a8985"
SCHEME_COLOURS = {"Exact": GREEN, "QE": BLUE, "Euler": ORANGE}

st.title("Monte Carlo Lab")
st.markdown(
    "A simulation is only as good as its time-stepping scheme. This page measures how wrong "
    "each scheme is, and why (book Ch 9). Every number comes from your C++ engine."
)

tab_gbm, tab_cir, tab_heston = st.tabs(
    ["Euler vs Milstein (GBM)", "Sampling the variance (CIR)", "Heston: fixing the bias"]
)

# ---------------- Tab 1: strong and weak convergence ----------------
with tab_gbm:
    st.latex(r"\text{Euler: } S_{i+1} = S_i + rS_i\Delta t + \sigma S_i\Delta W \qquad "
             r"\text{Milstein: } \ldots + \tfrac12\sigma^2 S_i\,(\Delta W^2 - \Delta t)")
    c1, c2, c3 = st.columns(3)
    sigma = c1.slider("Volatility σ", 0.1, 0.8, 0.3, step=0.05)
    r = c2.slider("Rate r", 0.0, 0.10, 0.06, step=0.01)
    n_paths = c3.select_slider("Paths", [5_000, 20_000, 50_000], value=20_000)
    S0, T, K = 100.0, 1.0, 100.0
    steps = np.array([4, 8, 16, 32, 64, 128, 256])
    dts = T / steps

    strong_eu, strong_mi, weak_eu, weak_mi = [], [], [], []
    exact_call = bs_price(OptionType.Call, S0, K, T, r, sigma)
    for n in steps:
        ex, eu, mi = gbm_schemes(S0, r, sigma, T, int(n), n_paths, 1)
        strong_eu.append(np.mean(np.abs(eu - ex)))
        strong_mi.append(np.mean(np.abs(mi - ex)))
        # weak error: compare each scheme's option price with the exact-solution MC price
        # on the SAME paths, so the statistical noise cancels and only the bias is left
        disc = math.exp(-r * T)
        ref = disc * np.maximum(ex - K, 0.0).mean()
        weak_eu.append(abs(disc * np.maximum(eu - K, 0.0).mean() - ref))
        weak_mi.append(abs(disc * np.maximum(mi - K, 0.0).mean() - ref))

    def order(e):
        return np.polyfit(np.log(dts), np.log(np.maximum(e, 1e-12)), 1)[0]

    left, right = st.columns(2)
    for col, title, e1, e2, guide_order in [
        (left, "Strong error E|S_T(scheme) − S_T(exact)|", strong_eu, strong_mi, None),
        (right, "Weak error |price(scheme) − price(exact)|", weak_eu, weak_mi, None),
    ]:
        fig = go.Figure()
        fig.add_scatter(x=dts, y=e1, name=f"Euler (order ≈ {order(e1):.2f})",
                        mode="lines+markers", line=dict(color=ORANGE, width=2), marker=dict(size=8))
        fig.add_scatter(x=dts, y=e2, name=f"Milstein (order ≈ {order(e2):.2f})",
                        mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8))
        fig.update_layout(title=title, xaxis_title="Time step Δt (log)", yaxis_title="Error (log)",
                          xaxis_type="log", yaxis_type="log", yaxis_exponentformat="power",
                          height=420, margin=dict(t=50), legend=dict(y=-0.3, orientation="h"))
        col.plotly_chart(fig, width="stretch")
    st.caption(
        "Strong error measures each path against the exact path driven by the same Brownian "
        "motion: Euler's falls like Δt^½ (halve it by taking 4× more steps), Milstein's like Δt¹ "
        "(2× more steps). That matters for hedging and path-dependent products. The weak error "
        "only asks whether averages are right: both schemes are order 1 there, so for a European "
        "price Euler is often good enough. Don't be surprised if Euler's weak error is even "
        "smaller than Milstein's here: both are order 1, and for this payoff Euler happens to have "
        "the smaller constant. Milstein's advantage is path by path. (Book Figs 9.4 and 9.7.)"
    )

# ---------------- Tab 2: one big CIR step, three ways ----------------
with tab_cir:
    st.markdown(
        "The Heston variance follows a CIR process. Take **one** step of size Δt from v₀ and "
        "look at the distribution of v(Δt) under each scheme. The exact answer is a scaled "
        "noncentral χ² (book eq. 9.30)."
    )
    case = st.radio("Test case (book Example 9.3.2)", [
        "γ = 0.2: far from zero (QE uses its quadratic branch)",
        "γ = 0.6: mass near zero (QE uses its exponential branch)"], horizontal=False)
    gamma_c = 0.2 if case.startswith("γ = 0.2") else 0.6
    c1, c2 = st.columns(2)
    dt_c = c1.slider("Step size Δt (years)", 0.05, 5.0, 5.0, step=0.05)
    n_c = c2.select_slider("Samples", [20_000, 100_000, 400_000], value=100_000)
    v0_c, kappa_c, vbar_c = 0.3, 0.5, 0.05
    ek = math.exp(-kappa_c * dt_c)
    m = vbar_c + (v0_c - vbar_c) * ek
    s2 = v0_c * gamma_c**2 * ek * (1 - ek) / kappa_c + vbar_c * gamma_c**2 * (1 - ek) ** 2 / (2 * kappa_c)
    st.caption(f"κ = {kappa_c}, v̄ = {vbar_c}, v₀ = {v0_c}. ψ = s²/m² = {s2 / m**2:.2f} "
               f"({'≤' if s2 / m**2 <= 1.5 else '>'} 1.5, so QE uses its "
               f"{'quadratic' if s2 / m**2 <= 1.5 else 'exponential'} branch).")

    samples = {name: cir_step_samples(sch, v0_c, kappa_c, vbar_c, gamma_c, dt_c, n_c, 2)
               for name, sch in [("Exact", CirScheme.Exact), ("QE", CirScheme.QE),
                                 ("Euler", CirScheme.Euler)]}
    rows = [{"Scheme": "Theory", "Mean": f"{m:.5f}", "Variance": f"{s2:.6f}", "P(v = 0)": "—"}]
    for name, x in samples.items():
        rows.append({"Scheme": name, "Mean": f"{x.mean():.5f}", "Variance": f"{x.var():.6f}",
                     "P(v = 0)": f"{(x == 0).mean():.1%}"})
    st.dataframe(rows, hide_index=True, width="stretch")

    view = st.radio("Show", ["CDF (like book Fig 9.12)", "Histogram"], horizontal=True)
    hi = float(np.percentile(samples["Exact"], 99.5))
    fig = go.Figure()
    for name, x in samples.items():
        if view.startswith("CDF"):
            xs = np.sort(x)[:: max(1, len(x) // 2000)]
            fig.add_scatter(x=xs, y=np.searchsorted(np.sort(x), xs, side="right") / len(x),
                            name=name, line=dict(color=SCHEME_COLOURS[name], width=2,
                                                 dash="dash" if name == "QE" else "solid"))
        else:
            fig.add_histogram(x=x[x <= hi], name=name, histnorm="probability density",
                              nbinsx=120, opacity=0.5, marker=dict(color=SCHEME_COLOURS[name]))
    fig.update_layout(xaxis_title="v(Δt)", yaxis_title="P(v ≤ x)" if view.startswith("CDF")
                      else "Density", xaxis_range=[0, hi], height=440,
                      barmode="overlay")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "QE lies on top of the exact CDF; Euler, taking the same single step, ends up with the "
        "wrong mean and a huge spike at 0 (it overshoots below zero and gets floored). QE's "
        "exponential branch also puts mass exactly at 0, but the right amount: that's how it "
        "matches the near-singular density when the Feller condition fails. Shrink Δt and "
        "Euler slowly gets better; QE and exact are good at any step size."
    )

# ---------------- Tab 3: Heston pricing bias by scheme and step size ----------------
with tab_heston:
    st.markdown(
        "Back to the failed check on the Heston page. Price an at-the-money call under Heston "
        "with three variance schemes, for different numbers of time steps, and compare with "
        "the exact COS price. The log-price always uses the almost-exact step (book eq. 9.63)."
    )
    preset = st.selectbox("Parameters", ["Fang & Oosterlee (Feller violated)",
                                         "Typical equity (Feller violated)",
                                         "Calm market (Feller satisfied)"])
    P = {"Fang & Oosterlee (Feller violated)": dict(kappa=1.5768, vbar=0.0398, gamma=0.5751,
                                                     rho=-0.5711, v0=0.0175),
         "Typical equity (Feller violated)": dict(kappa=1.5, vbar=0.04, gamma=0.6, rho=-0.7,
                                                   v0=0.04),
         "Calm market (Feller satisfied)": dict(kappa=3.0, vbar=0.04, gamma=0.3, rho=-0.7,
                                                v0=0.04)}[preset]
    c1, c2 = st.columns(2)
    T_h = c1.slider("Maturity T (years)", 0.25, 3.0, 1.0, step=0.25)
    n_h = c2.select_slider("Paths per point", [20_000, 50_000, 100_000, 200_000], value=50_000)
    feller = 2 * P["kappa"] * P["vbar"] >= P["gamma"] ** 2
    st.caption(f"2κv̄ = {2 * P['kappa'] * P['vbar']:.3f} vs γ² = {P['gamma'] ** 2:.3f}: "
               f"Feller {'satisfied' if feller else 'violated'}.")

    r_h = 0.0
    ref = cos_price(HestonModel(r=r_h, **P), OptionType.Call, 100.0, 100.0, T_h, N=256, L=12.0)
    step_list = [2, 4, 8, 16, 32, 64]

    @st.cache_data(show_spinner="Running Heston simulations in C++...")
    def bias_table(P_items, T_h, n_h):
        P_ = dict(P_items)
        out = {}
        for name, sch in [("Euler", CirScheme.Euler), ("Exact", CirScheme.Exact),
                          ("QE", CirScheme.QE)]:
            vals = []
            for n in step_list:
                ST = heston_terminal(sch, 100.0, r_h, P_["kappa"], P_["vbar"], P_["gamma"],
                                     P_["rho"], P_["v0"], T_h, n, n_h, 7)
                pay = math.exp(-r_h * T_h) * np.maximum(ST - 100.0, 0.0)
                vals.append((pay.mean(), pay.std(ddof=1) / math.sqrt(n_h)))
            out[name] = vals
        return out

    res = bias_table(tuple(sorted(P.items())), T_h, n_h)
    fig = go.Figure()
    band = 1.96 * np.mean([se for vals in res.values() for _, se in vals])
    fig.add_scatter(x=step_list + step_list[::-1], y=[band] * len(step_list) + [-band] * len(step_list),
                    fill="toself", fillcolor="rgba(138, 137, 133, 0.15)", line=dict(width=0),
                    mode="lines", name="± Monte Carlo noise (95%)", hoverinfo="skip")
    for name, vals in res.items():
        fig.add_scatter(x=step_list, y=[p - ref for p, _ in vals], name=name, mode="lines+markers",
                        line=dict(color=SCHEME_COLOURS[name], width=2), marker=dict(size=9))
    fig.add_hline(y=0.0, line=dict(color=NEUTRAL, width=1, dash="dot"))
    fig.update_layout(xaxis_title="Time steps per path (log scale)", xaxis_type="log",
                      xaxis=dict(tickvals=step_list, ticktext=[str(n) for n in step_list]),
                      yaxis_title="MC price − exact COS price", height=450)
    st.plotly_chart(fig, width="stretch")

    rows = []
    for i, n in enumerate(step_list):
        row = {"Steps": n}
        for name in ["Euler", "Exact", "QE"]:
            p, se = res[name][i]
            row[name] = f"{p - ref:+.4f}" + ("" if abs(p - ref) < 1.96 * se else "  ✗")
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        f"Exact COS price: {ref:.6f}. ✗ marks an error larger than the Monte Carlo noise, i.e. "
        "real bias. Euler's bias shrinks only slowly with more steps, and is worst when the "
        "Feller condition is violated. Exact and QE variance sampling are inside the noise band "
        "from about 8–16 steps. Try the 'Calm market' preset: with Feller satisfied, Euler is "
        "much better behaved. One honest caveat: with very few steps, AES/QE is not exactly a "
        "martingale; Andersen (2008) adds a 'martingale correction' for that."
    )