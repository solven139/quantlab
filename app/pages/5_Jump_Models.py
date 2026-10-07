import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantlab import (BlackScholesModel, KouModel, MertonModel, OptionType,
                      VarianceGammaModel, bs_price, cos_density, cos_price, implied_vol,
                      simulate_merton)

BLUE = "#3987e5"
ORANGE = "#d95926"
NEUTRAL = "#8a8985"
GREEN = "#1baf7a"
PATH_COLOR = "rgba(57, 135, 229, 0.35)"

st.title("Jump Models")
st.markdown(
    "Real returns have **fat tails** (crashes are far more common than a normal distribution "
    "allows) and **skew** (big drops are more common than big rises). Black-Scholes has "
    "neither, so it can't produce the implied-volatility smile seen in markets. "
    "Adding jumps fixes that (book Ch 5)."
)

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Model")
    name = st.radio("Jump model", ["Merton", "Kou", "Variance Gamma"])
    S0 = st.number_input("Spot S0", value=100.0, min_value=1.0)
    r = st.slider("Rate r", 0.0, 0.10, 0.03, step=0.005)

    if name == "Merton":
        st.caption("GBM + Poisson jumps, each multiplying S by e^J with J ~ N(μ_J, σ_J²).")
        sigma = st.slider("Diffusion volatility σ", 0.05, 0.50, 0.15, step=0.01)
        xi = st.slider("Jump intensity ξ (jumps per year)", 0.0, 3.0, 0.5, step=0.1)
        mu_j = st.slider("Mean jump size μ_J", -0.40, 0.20, -0.10, step=0.01)
        sigma_j = st.slider("Jump size volatility σ_J", 0.01, 0.40, 0.15, step=0.01)
        model = MertonModel(r=r, sigma=sigma, xi=xi, mu_j=mu_j, sigma_j=sigma_j)
        sde = r"\frac{dS}{S} = (r-\bar\omega)\,dt + \sigma\,dW + (e^{J}-1)\,dX_P,\quad J\sim\mathcal N(\mu_J,\sigma_J^2)"
    elif name == "Kou":
        st.caption("GBM + jumps that are up (Exp(η₁)) with probability p, else down (−Exp(η₂)).")
        sigma = st.slider("Diffusion volatility σ", 0.05, 0.50, 0.15, step=0.01)
        xi = st.slider("Jump intensity ξ (jumps per year)", 0.0, 3.0, 0.8, step=0.1)
        p = st.slider("Probability a jump is upward p", 0.0, 1.0, 0.35, step=0.05)
        eta1 = st.slider("Up-jump decay η₁ (mean up-jump = 1/η₁)", 1.5, 50.0, 12.0, step=0.5)
        eta2 = st.slider("Down-jump decay η₂ (mean down-jump = 1/η₂)", 1.0, 50.0, 8.0, step=0.5)
        model = KouModel(r=r, sigma=sigma, xi=xi, p=p, eta1=eta1, eta2=eta2)
        sde = r"\frac{dS}{S} = (r-\bar\omega)\,dt + \sigma\,dW + (e^{J}-1)\,dX_P,\quad J\sim\text{double exponential}"
    else:
        st.caption("Brownian motion with drift θ, run on a random Gamma clock with variance rate β.")
        sigma = st.slider("Volatility σ", 0.05, 0.50, 0.20, step=0.01)
        theta = st.slider("Drift of the subordinated BM θ (skew)", -0.50, 0.30, -0.15, step=0.01)
        beta = st.slider("Variance rate of the clock β (kurtosis)", 0.01, 1.0, 0.20, step=0.01)
        try:
            model = VarianceGammaModel(r=r, sigma=sigma, theta=theta, beta=beta)
        except ValueError:
            st.error("These parameters make the drift correction undefined "
                     "(need θ + σ²/2 < 1/β). Lower θ or β.")
            st.stop()
        sde = r"X(t) = (r+\bar\omega)\,t + \theta\,G(t) + \sigma\,W(G(t)),\quad G(t)\sim\Gamma"

st.latex(sde)
omega = model.drift_correction()
st.caption(
    f"Drift correction ω̄ = {omega:+.5f}: chosen so that E[S(T)] = S0·e^(rT), i.e. the "
    "discounted stock stays a martingale and there's no arbitrage. "
    + ("With mostly downward jumps, ω̄ < 0, so the stock drifts up a little faster between "
       "jumps to make up for the expected crashes." if omega < 0 and name != "Variance Gamma"
       else "")
)

tab_paths, tab_dist, tab_smile, tab_check = st.tabs(
    ["Sample paths", "Fat tails", "Volatility smile", "Pricing checks"]
)

# ---------------- Tab 1: paths with jumps ----------------
with tab_paths:
    c1, c2 = st.columns(2)
    T_paths = c1.slider("Horizon (years)", 0.5, 5.0, 2.0, step=0.5)
    n_draw = c2.slider("Paths drawn", 5, 100, 25)
    n_steps = 500
    t = np.linspace(0.0, T_paths, n_steps + 1)
    rng = np.random.default_rng(7)
    dt = T_paths / n_steps

    if name == "Merton":
        paths = simulate_merton(S0, r, sigma, xi, mu_j, sigma_j, T_paths, n_steps, n_draw, 7)
    elif name == "Kou":
        # Same idea as the C++ Merton simulator, written here in NumPy for the picture only
        dX = (r - omega - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal((n_draw, n_steps))
        jumps = rng.poisson(xi * dt, (n_draw, n_steps))
        up = rng.random((n_draw, n_steps)) < p
        size = np.where(up, rng.exponential(1 / eta1, (n_draw, n_steps)),
                        -rng.exponential(1 / eta2, (n_draw, n_steps)))
        dX += jumps * size  # at most one jump per step is a good approximation for small dt
        paths = S0 * np.exp(np.hstack([np.zeros((n_draw, 1)), np.cumsum(dX, axis=1)]))
    else:
        dG = rng.gamma(dt / beta, beta, (n_draw, n_steps))       # random business time
        dX = (r + omega) * dt + theta * dG + sigma * np.sqrt(dG) * rng.standard_normal((n_draw, n_steps))
        paths = S0 * np.exp(np.hstack([np.zeros((n_draw, 1)), np.cumsum(dX, axis=1)]))

    fig = go.Figure()
    for row in paths:
        fig.add_scatter(x=t, y=row, mode="lines", line=dict(color=PATH_COLOR, width=1),
                        showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=t, y=S0 * np.exp(r * t), name="E[S(t)] = S0 e^(rt)",
                    line=dict(color=ORANGE, width=3))
    fig.update_layout(xaxis_title="Time t (years)", yaxis_title="S(t)", height=460)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        {"Merton": "The vertical drops are jumps: the price gaps down in an instant, which "
                   "no continuous hedge can follow. Raise ξ for more of them, lower μ_J for bigger crashes.",
         "Kou": "Jumps both ways, but with asymmetric sizes. Try η₂ small (big down-jumps) "
                "and p small (mostly down).",
         "Variance Gamma": "No smooth diffusion at all: the path is made of infinitely many "
                           "small jumps. Larger β makes the clock more irregular: calm periods "
                           "and bursts."}[name]
    )

# ---------------- Tab 2: the return distribution vs a normal with the same variance ----------------
with tab_dist:
    T_d = st.slider("Horizon T (years)", 0.1, 3.0, 1.0, step=0.1, key="T_dist")
    c1, c2, c4 = model.cumulant1(T_d), model.cumulant2(T_d), model.cumulant4(T_d)
    sd = math.sqrt(c2)
    xs = np.linspace(c1 - 6 * sd, c1 + 6 * sd, 500)
    f_model = np.maximum(np.array(cos_density(model, T_d, list(xs), N=512, L=12.0)), 1e-300)
    f_norm = np.exp(-0.5 * ((xs - c1) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

    m1, m2, m3 = st.columns(3)
    m1.metric("Std of log-return", f"{sd:.2%}")
    m2.metric("Excess kurtosis", f"{c4 / c2**2:.2f}", help="0 for a normal distribution")
    m3.metric("P(return < −4 std)", f"{np.trapezoid(f_model[xs < c1 - 4 * sd], xs[xs < c1 - 4 * sd]):.2e}",
              help="A normal distribution gives 3.2e-05")

    log_y = st.toggle("Log scale (shows the tails)", value=True)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=f_model, name=f"{name} (from φ via COS)",
                    line=dict(color=BLUE, width=2))
    fig.add_scatter(x=xs, y=f_norm, name="Normal, same mean and variance",
                    line=dict(color=ORANGE, width=2, dash="dash"))
    fig.update_layout(xaxis_title="Log-return X = log(S(T)/S0)", yaxis_title="Density",
                      yaxis_type="log" if log_y else "linear", height=450)
    if log_y:
        fig.update_yaxes(range=[-6, math.log10(max(f_model.max(), f_norm.max())) + 0.2],
                         exponentformat="power")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "On a log scale a normal density is a parabola, so its tails die off extremely fast. "
        "The jump model's tails are much higher: extreme moves are orders of magnitude more "
        "likely. Same variance, very different risk. This density was recovered from the "
        "characteristic function with your COS code; no formula for it was needed."
    )

# ---------------- Tab 3: the smile these prices imply ----------------
with tab_smile:
    st.markdown(
        "Price options at many strikes with your COS pricer, then invert each price with your "
        "implied-vol solver. Black-Scholes would give a flat line."
    )
    mny = np.linspace(0.6, 1.4, 33)
    fig = go.Figure()
    for colour, T_s in zip([BLUE, ORANGE, GREEN], [0.25, 1.0, 2.0]):
        F = S0 * math.exp(r * T_s)
        ks, ivs = [], []
        for m_ in mny:
            k = m_ * F
            typ = OptionType.Put if k < F else OptionType.Call
            try:
                v = cos_price(model, typ, S0, k, T_s, N=512, L=12.0)
                ivs.append(implied_vol(typ, v, S0, k, T_s, r))
                ks.append(m_)
            except ValueError:
                pass
        fig.add_scatter(x=ks, y=ivs, name=f"T = {T_s:g} years", mode="lines+markers",
                        line=dict(color=colour, width=2), marker=dict(size=6))
    fig.update_layout(xaxis_title="Moneyness K / forward", yaxis_title="Implied volatility",
                      yaxis_tickformat=".1%", height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Downward jumps make low-strike puts expensive (crash insurance), so implied vol rises "
        "to the left: a skew, like equity index markets. The effect is strongest at short "
        "maturities and fades for longer ones, because over a long horizon many jumps average "
        "out towards normal (central limit theorem). That fading is a known weakness of jump "
        "models; Heston (Step 5) behaves differently."
    )

    st.subheader("The full implied-volatility surface")
    mats = np.linspace(0.1, 3.0, 20)
    grid = np.linspace(0.6, 1.4, 25)
    Z = np.full((len(mats), len(grid)), np.nan)
    for i_, T_s in enumerate(mats):
        F = S0 * math.exp(r * T_s)
        for j_, m_ in enumerate(grid):
            k = m_ * F
            typ = OptionType.Put if k < F else OptionType.Call
            try:
                v = cos_price(model, typ, S0, k, T_s, N=256, L=12.0)
                Z[i_, j_] = implied_vol(typ, v, S0, k, T_s, r)
            except ValueError:
                pass
    fig = go.Figure(go.Surface(
        x=np.tile(grid, (len(mats), 1)), y=np.repeat(mats[:, None], len(grid), axis=1), z=Z,
        colorscale="Blues", colorbar=dict(title="Implied vol", tickformat=".0%", len=0.7),
        hovertemplate="K/F: %{x:.2f}<br>T: %{y:.2f} y<br>Implied vol: %{z:.2%}<extra></extra>"))
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(xaxis_title="Moneyness K / forward",
                                 yaxis_title="Maturity T (years)", zaxis_title="Implied vol",
                                 zaxis_tickformat=".0%",
                                 camera=dict(eye=dict(x=0.4, y=-2.1, z=0.8))))
    st.plotly_chart(fig, width="stretch")

# ---------------- Tab 4: is the COS price right? ----------------
with tab_check:
    T_c = st.slider("Maturity T (years)", 0.1, 3.0, 1.0, step=0.1, key="T_check")
    st.markdown(
        "Every number on this page comes from COS. Here it is checked against an independent "
        "method: Merton's 1976 closed-form series for Merton, and a fresh Monte Carlo "
        "simulation for Kou and Variance Gamma."
    )
    strikes = [0.8 * S0, 0.9 * S0, S0, 1.1 * S0, 1.2 * S0]
    rows = []
    if name == "Merton":
        k_ = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        lam = xi * (1.0 + k_)
        for K in strikes:
            ref = 0.0
            for n in range(80):
                sig_n = math.sqrt(sigma**2 + n * sigma_j**2 / T_c)
                r_n = r - xi * k_ + n * math.log(1.0 + k_) / T_c
                w = math.exp(-lam * T_c) * (lam * T_c) ** n / math.factorial(n)
                ref += w * bs_price(OptionType.Call, S0, K, T_c, r_n, sig_n)
            v = cos_price(model, OptionType.Call, S0, K, T_c, N=256, L=12.0)
            rows.append({"Strike": f"{K:g}", "COS": f"{v:.10f}", "Merton series": f"{ref:.10f}",
                         "Difference": f"{abs(v - ref):.1e}"})
    else:
        rng = np.random.default_rng(11)
        n = 400_000
        if name == "Kou":
            X = (r - omega - 0.5 * sigma**2) * T_c + sigma * math.sqrt(T_c) * rng.standard_normal(n)
            nj = rng.poisson(xi * T_c, n)
            n_up = rng.binomial(nj, p)
            n_dn = nj - n_up
            X += np.where(n_up > 0, rng.gamma(np.maximum(n_up, 1), 1.0 / eta1), 0.0)
            X -= np.where(n_dn > 0, rng.gamma(np.maximum(n_dn, 1), 1.0 / eta2), 0.0)
        else:
            G = rng.gamma(T_c / beta, beta, n)
            X = (r + omega) * T_c + theta * G + sigma * np.sqrt(G) * rng.standard_normal(n)
        ST = S0 * np.exp(X)
        for K in strikes:
            pay = math.exp(-r * T_c) * np.maximum(ST - K, 0.0)
            se = pay.std(ddof=1) / math.sqrt(n)
            v = cos_price(model, OptionType.Call, S0, K, T_c, N=256, L=12.0)
            rows.append({"Strike": f"{K:g}", "COS": f"{v:.6f}",
                         "Monte Carlo (400k paths)": f"{pay.mean():.6f} ± {1.96 * se:.4f}",
                         "Within 95% CI?": "yes" if abs(v - pay.mean()) < 1.96 * se else "no"})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "The same cos_price function priced Black-Scholes in Step 3. It never changed: it only "
        "sees the Model interface, so each new model is just a new characteristic function."
    )