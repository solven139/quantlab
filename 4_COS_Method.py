import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantlab import BlackScholesModel, OptionType, bs_price, cos_density, cos_price, simulate_gbm

BLUE = "#3987e5"
ORANGE = "#d95926"
NEUTRAL = "#8a8985"
# Light-to-dark blue for "more terms"; the exact answer is drawn in orange
RAMP = ["#b7d3f6", "#6da7ec", "#2a78d6", "#104281"]

st.title("The COS Method")
st.markdown(
    "Pricing an option means integrating its payoff against a density. For many models "
    "(jumps, Variance Gamma, Heston) the density is unknown, but the **characteristic "
    "function** φ(u) = E[e^{iuX}] is known in closed form. The COS method (book Ch 6) "
    "expands the density in a cosine series whose coefficients come straight from φ:"
)
st.latex(r"V \approx e^{-rT}\sum_{k=0}^{N-1}{}' \,\mathrm{Re}\Big\{\varphi(u_k)\,"
         r"e^{\,i u_k (x-a)}\Big\} H_k,\qquad u_k=\frac{k\pi}{b-a},\quad x=\log\frac{S_0}{K}")
st.caption(
    "Here the model is Black-Scholes, so every COS number can be checked against the exact "
    "formula. From Step 4 on, the same C++ pricer runs unchanged on models with no closed form."
)

with st.sidebar:
    st.header("Market")
    S0 = st.slider("Spot S0", 50.0, 150.0, 100.0)
    K = st.slider("Strike K", 50.0, 150.0, 100.0)
    T = st.slider("Maturity T (years)", 0.05, 5.0, 1.0)
    r = st.slider("Rate r", 0.0, 0.10, 0.05, step=0.005)
    sigma = st.slider("Volatility sigma", 0.05, 0.80, 0.25, step=0.01)

model = BlackScholesModel(r=r, sigma=sigma)
exact = bs_price(OptionType.Call, S0, K, T, r, sigma)

tab_dens, tab_conv, tab_mc, tab_L = st.tabs(
    ["Density from φ", "Convergence in N", "COS vs Monte Carlo", "Integration range L"]
)

# ---------------- Tab 1: recover the density from the characteristic function ----------------
with tab_dens:
    st.markdown(
        "Book Figure 6.2: the density of X = log(S(T)/S0) rebuilt from φ alone, with more and "
        "more cosine terms. With too few terms you get wiggles and even negative 'probabilities'."
    )
    space = st.radio("Show the density of", ["log-return X = log(S(T)/S0)", "price S(T)"],
                     horizontal=True)
    Ns = [4, 8, 16, 64]
    mu, s = (r - 0.5 * sigma**2) * T, sigma * np.sqrt(T)
    xs = np.linspace(mu - 5 * s, mu + 5 * s, 400)
    exact_pdf = np.exp(-0.5 * ((xs - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    if space.startswith("price"):
        xplot = S0 * np.exp(xs)
        jac = 1.0 / xplot           # change of variables: f_S(s) = f_X(log(s/S0)) / s
        xtitle = "S(T)"
    else:
        xplot = xs
        jac = np.ones_like(xs)
        xtitle = "X = log(S(T)/S0)"

    fig = go.Figure()
    for n, colour in zip(Ns, RAMP):
        f = np.array(cos_density(model, T, list(xs), N=n)) * jac
        fig.add_scatter(x=xplot, y=f, name=f"COS, N = {n}", line=dict(color=colour, width=2))
    fig.add_scatter(x=xplot, y=exact_pdf * jac, name="Exact density",
                    line=dict(color=ORANGE, width=2, dash="dash"))
    fig.add_hline(y=0.0, line=dict(color=NEUTRAL, width=1, dash="dot"))
    fig.update_layout(xaxis_title=xtitle, yaxis_title="Density", height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "N = 64 is indistinguishable from the exact curve. Every coefficient came from φ(u_k), "
        "so the same picture works for any model that has a characteristic function."
    )

# ---------------- Tab 2: exponential convergence ----------------
with tab_conv:
    Ns = np.array([2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    strikes = sorted({round(0.8 * S0, 2), round(S0, 2), round(1.2 * S0, 2), round(K, 2)})
    fig = go.Figure()
    palette = [BLUE, ORANGE, "#1baf7a", "#4a3aa7"]
    for colour, k in zip(palette, strikes):
        ref = bs_price(OptionType.Call, S0, k, T, r, sigma)
        err = [max(abs(cos_price(model, OptionType.Call, S0, k, T, N=int(n)) - ref), 1e-16)
               for n in Ns]
        fig.add_scatter(x=Ns, y=err, name=f"K = {k:g}", mode="lines+markers",
                        line=dict(color=colour, width=2), marker=dict(size=8))
    fig.add_scatter(x=[Ns[0], Ns[-1]], y=[1e-15, 1e-15], mode="lines",
                    name="machine precision (~1e-15)",
                    line=dict(color=NEUTRAL, width=1, dash="dot"), hoverinfo="skip")
    fig.update_layout(xaxis_title="Number of cosine terms N",
                      yaxis_title="|COS − Black-Scholes| (log scale)", yaxis_type="log",
                      yaxis_exponentformat="power", height=450)
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Black-Scholes formula", f"{exact:.8f}")
    c2.metric("COS, N = 64", f"{cos_price(model, OptionType.Call, S0, K, T, N=64):.8f}")
    c3.metric("Difference", f"{abs(cos_price(model, OptionType.Call, S0, K, T, N=64) - exact):.1e}")
    st.caption(
        "On a log scale the error falls along a straight line, then drops to machine "
        "precision: exponential convergence (book §6.2.3), because the normal density is "
        "infinitely smooth. Short maturities need more terms: the density is narrower, so its "
        "cosine series has more high-frequency content."
    )

# ---------------- Tab 3: speed and accuracy against Monte Carlo ----------------
with tab_mc:
    st.markdown("The same call priced two ways. Both use your C++ engine.")
    n_paths = st.select_slider("Monte Carlo paths", [10_000, 50_000, 100_000, 500_000],
                               value=100_000)

    t0 = time.perf_counter()
    for _ in range(100):
        v_cos = cos_price(model, OptionType.Call, S0, K, T, N=64)
    t_cos = (time.perf_counter() - t0) / 100

    t0 = time.perf_counter()
    ST = simulate_gbm(S0, r, sigma, T, 1, n_paths, 42)[:, -1]
    disc = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    v_mc = disc.mean()
    t_mc = time.perf_counter() - t0
    se = disc.std(ddof=1) / np.sqrt(n_paths)

    c1, c2 = st.columns(2)
    c1.metric("COS (N = 64)", f"{v_cos:.6f}",
              help="Error vs the exact formula shown below")
    c1.caption(f"error {abs(v_cos - exact):.1e} · time {t_cos * 1e6:.0f} µs")
    c2.metric(f"Monte Carlo ({n_paths:,} paths)", f"{v_mc:.6f}")
    c2.caption(f"error {abs(v_mc - exact):.1e} (± {1.96 * se:.1e} at 95%) · "
               f"time {t_mc * 1e3:.1f} ms")
    st.success(f"COS is about {t_mc / max(t_cos, 1e-9):,.0f}× faster here, and its error "
               f"({abs(v_cos - exact):.0e}) is at rounding level, while Monte Carlo is off by "
               f"{abs(v_mc - exact):.0e}. To gain one more digit, Monte Carlo needs 100× "
               "more paths; COS needs a few more terms.")
    st.caption(
        "That's why the book uses COS for calibration, where thousands of European prices are "
        "needed, and keeps Monte Carlo for path-dependent products and complex dynamics "
        "(Ch 9) where no characteristic function is available."
    )

# ---------------- Tab 4: the truncation range ----------------
with tab_L:
    st.markdown(
        "COS cuts the integral to [a, b] = mean ± L standard deviations (book eq. 6.44). "
        "Too narrow loses probability mass; very wide needs more terms. Book Figure 6.3 "
        "suggests L = 8."
    )
    N_fixed = st.select_slider("Cosine terms N", [32, 64, 128, 256, 1024], value=256)
    Ls = np.arange(1.0, 12.5, 0.5)
    fig = go.Figure()
    for colour, k in zip([BLUE, ORANGE, "#1baf7a"], [0.8 * S0, S0, 1.2 * S0]):
        ref = bs_price(OptionType.Call, S0, k, T, r, sigma)
        err = [max(abs(cos_price(model, OptionType.Call, S0, k, T, N=N_fixed, L=float(L)) - ref),
                   1e-16) for L in Ls]
        fig.add_scatter(x=Ls, y=err, name=f"K = {k:g}", mode="lines+markers",
                        line=dict(color=colour, width=2), marker=dict(size=7))
    fig.add_vline(x=8.0, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="L = 8")
    fig.update_layout(xaxis_title="L (half-width of [a, b] in standard deviations)",
                      yaxis_title="Absolute error (log scale)", yaxis_type="log",
                      yaxis_exponentformat="power", height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Error falls exponentially as L grows, because the normal tails decay like e^(−L²/2). "
        "Try N = 32 with a large L: the range becomes too wide for that few terms and the "
        "error rises again. That's the trade-off between the two error sources."
    )
