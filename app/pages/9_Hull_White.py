import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantlab import HullWhiteModel, NelsonSiegelCurve, simulate_hull_white

BLUE = "#3987e5"
ORANGE = "#d95926"
GREEN = "#1baf7a"
VIOLET = "#4a3aa7"
NEUTRAL = "#8a8985"
# Light-to-dark blue: paths coloured by where the short rate ends
RAMP = ["#b7d3f6", "#6da7ec", "#2a78d6", "#104281"]

st.title("Hull-White Short Rate")
st.markdown(
    "Until now the interest rate was a constant r. In reality it moves, and bonds, swaps and "
    "every discount factor depend on how. The Hull-White model (book Ch 11.3) lets the "
    "**short rate** r(t), the rate for borrowing over the next instant, follow a "
    "mean-reverting random process:"
)
st.latex(r"dr(t) = \lambda\big(\theta(t) - r(t)\big)\,dt + \eta\,dW(t),\qquad "
         r"P(0,T) = \mathbb{E}\Big[e^{-\int_0^T r(s)\,ds}\Big]")
st.caption(
    "λ pulls r back towards θ(t), η sets how much it jiggles. The function θ(t) is chosen so "
    "that the model reproduces today's yield curve P(0, T) exactly (book eq. 11.37). That fit "
    "is what makes Hull-White usable for pricing: it starts from the market."
)

PRESETS = {
    "Upward sloping (normal)": (0.04, -0.02, 0.01, 2.0),
    "Inverted": (0.03, 0.02, -0.01, 1.5),
    "Humped": (0.035, -0.01, 0.05, 1.5),
    "Flat": (0.03, 0.0, 0.0, 2.0),
}

with st.sidebar:
    st.header("Today's yield curve")
    st.caption("Nelson-Siegel form: level, slope, hump.")
    preset = st.selectbox("Start from", list(PRESETS))
    p = PRESETS[preset]
    b0 = st.slider("β₀  long-run level", 0.0, 0.08, p[0], step=0.0025, format="%.4f",
                   key=f"b0{preset}")
    b1 = st.slider("β₁  slope (short − long)", -0.05, 0.05, p[1], step=0.0025,
                   format="%.4f", key=f"b1{preset}")
    b2 = st.slider("β₂  hump", -0.05, 0.08, p[2], step=0.0025, format="%.4f",
                   key=f"b2{preset}")
    tau = st.slider("τ  where the hump sits (years)", 0.5, 5.0, p[3], step=0.25,
                    key=f"tau{preset}")
    st.header("Hull-White dynamics")
    lam = st.slider("λ  mean-reversion speed", 0.02, 2.0, 0.5, step=0.02)
    eta = st.slider("η  volatility of r", 0.002, 0.03, 0.01, step=0.001, format="%.3f")
    st.caption(f"Long-run std of r: η/√(2λ) = {eta / np.sqrt(2 * lam):.2%}")

curve = NelsonSiegelCurve(b0=b0, b1=b1, b2=b2, tau=tau)
hw = HullWhiteModel(lambda_=lam, eta=eta, curve=curve)
HORIZON, N_STEPS = 10.0, 400
times = np.linspace(0.0, HORIZON, N_STEPS + 1)


@st.cache_data(show_spinner="Simulating Hull-White paths in C++...")
def cached_paths(b0, b1, b2, tau, lam, eta, n_paths, seed):
    m = HullWhiteModel(lambda_=lam, eta=eta, curve=NelsonSiegelCurve(b0, b1, b2, tau))
    return simulate_hull_white(m, HORIZON, N_STEPS, n_paths, seed)


def lines_trace(x_times, rows, color, name):
    # Many paths in ONE trace, separated by NaN gaps
    x = np.tile(np.append(x_times, np.nan), len(rows))
    y = np.hstack([rows, np.full((len(rows), 1), np.nan)]).ravel()
    return go.Scattergl(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=1),
                        opacity=0.55, hoverinfo="skip", connectgaps=False)


tab_curve, tab_paths, tab_fit, tab_future = st.tabs(
    ["Today's curve and θ(t)", "Short-rate paths", "Fitting the curve", "Future yield curves"]
)

# ---------------- Tab 1: the curve, forwards and theta ----------------
with tab_curve:
    Ts = np.linspace(0.0, 15.0, 301)
    fig = go.Figure()
    fig.add_scatter(x=Ts, y=[curve.zero_rate(T) for T in Ts], name="Zero rate y(0, T)",
                    line=dict(color=BLUE, width=2.5))
    fig.add_scatter(x=Ts, y=[curve.inst_forward(T) for T in Ts],
                    name="Instantaneous forward f(0, T)", line=dict(color=GREEN, width=2.5))
    fig.add_scatter(x=Ts, y=[hw.theta(T) for T in Ts], name="Hull-White θ(t)",
                    line=dict(color=ORANGE, width=2.5, dash="dash"))
    fig.update_layout(xaxis_title="Maturity T / time t (years)", yaxis_title="Rate",
                      yaxis_tickformat=".2%", height=450)
    st.plotly_chart(fig, width="stretch")
    st.latex(r"\theta(t) = f(0,t) + \frac{1}{\lambda}\frac{\partial f(0,t)}{\partial t} "
             r"+ \frac{\eta^2}{2\lambda^2}\big(1 - e^{-2\lambda t}\big)")
    st.caption(
        "The zero rate y is the average yield up to T; the forward f is the rate the curve "
        "implies for a loan starting at T, so y is the running average of f. θ(t) is where the "
        "short rate is being pulled. It leads the forward curve: where f is still rising, θ is "
        "above it (the 1/λ · ∂f/∂t term), so that r arrives at the forward level on time. "
        "Make λ small and θ overshoots wildly; make it large and θ hugs f. The last term is a "
        "small convexity correction from the volatility."
    )

# ---------------- Tab 2: paths ----------------
with tab_paths:
    c1, c2, c3 = st.columns(3)
    n_sim = c1.select_slider("Paths simulated", [2_000, 10_000, 25_000], value=10_000)
    n_draw = c2.slider("Paths drawn", 10, 500, 150, step=10)
    seed = c3.number_input("Random seed", value=42, step=1)
    r_paths, I_paths = cached_paths(b0, b1, b2, tau, lam, eta, n_sim, int(seed))

    shown = r_paths[:n_draw]
    edges = np.quantile(shown[:, -1], np.linspace(0, 1, len(RAMP) + 1))
    bucket = np.clip(np.searchsorted(edges, shown[:, -1], side="right") - 1, 0, len(RAMP) - 1)
    fig = go.Figure()
    for b, colour in enumerate(RAMP):
        fig.add_trace(lines_trace(times, shown[bucket == b], colour,
                                  f"r(10y) {edges[b]:.1%} to {edges[b + 1]:.1%}"))
    mean = np.array([hw.mean_r(t) for t in times])
    sd = np.sqrt([hw.var_r(t) for t in times])
    fig.add_trace(go.Scattergl(x=times, y=mean + 2 * sd, mode="lines", name="E[r(t)] ± 2 sd",
                               line=dict(color=ORANGE, width=2, dash="dot")))
    fig.add_trace(go.Scattergl(x=times, y=mean - 2 * sd, mode="lines", showlegend=False,
                               line=dict(color=ORANGE, width=2, dash="dot")))
    fig.add_trace(go.Scattergl(x=times, y=mean, mode="lines", name="E[r(t)] = ψ(t)",
                               line=dict(color=ORANGE, width=4)))
    fig.add_hline(y=0.0, line=dict(color=NEUTRAL, width=1))
    fig.update_layout(xaxis_title="Time t (years)", yaxis_title="Short rate r(t)",
                      yaxis_tickformat=".1%", height=460, legend=dict(itemsizing="constant"))
    st.plotly_chart(fig, width="stretch")

    t_h = st.select_slider("Distribution of r at time t", [1.0, 2.0, 5.0, 10.0], value=5.0)
    j = int(round(t_h / HORIZON * N_STEPS))
    rt = r_paths[:, j]
    m, s = hw.mean_r(t_h), np.sqrt(hw.var_r(t_h))
    grid = np.linspace(m - 4 * s, m + 4 * s, 300)
    fig = go.Figure()
    fig.add_histogram(x=rt, histnorm="probability density", nbinsx=80,
                      name=f"Simulated r({t_h:g}y)", marker_color=BLUE, opacity=0.6)
    fig.add_scatter(x=grid, y=np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)),
                    name="Exact normal density", line=dict(color=ORANGE, width=2.5))
    fig.add_vline(x=0.0, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="r = 0")
    fig.update_layout(xaxis_title="r", xaxis_tickformat=".1%", yaxis_title="Density",
                      height=360, bargap=0.02)
    st.plotly_chart(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean of r (sim / exact)", f"{rt.mean():.3%}", f"exact {m:.3%}",
              delta_color="off")
    c2.metric("Std of r (sim / exact)", f"{rt.std():.3%}", f"exact {s:.3%}", delta_color="off")
    c3.metric("P(r < 0)", f"{(rt < 0).mean():.1%}")
    st.caption(
        "r(t) is exactly normal, because the model is linear in r with additive noise. That "
        "makes bond prices closed-form, but it also means negative rates have positive "
        "probability. Raise η or lower β₀ to see it grow. The variance saturates at "
        "η²/(2λ): mean reversion stops the fan from widening forever, unlike a stock."
    )

# ---------------- Tab 3: fit to today's curve ----------------
with tab_fit:
    st.markdown(
        "The test the whole construction is built for: discount along every simulated path, "
        "average, and you should land on today's market prices P(0, T). For comparison, the "
        "**Vasicek** model uses a constant target θ̄ (here the long rate β₀) and has no way to "
        "match an arbitrary curve."
    )
    r_paths, I_paths = cached_paths(b0, b1, b2, tau, lam, eta, 25_000, 42)
    Tg = np.arange(0.5, HORIZON + 1e-9, 0.5)
    idx = np.round(Tg / HORIZON * N_STEPS).astype(int)
    D = np.exp(-I_paths[:, idx])
    P_mc, se = D.mean(axis=0), D.std(axis=0, ddof=1) / np.sqrt(len(D))
    P_mkt = np.array([curve.discount(T) for T in Tg])

    # Vasicek closed form with constant theta_bar (book Ch 11.2)
    th, r0 = b0, hw.r0()
    B = (1 - np.exp(-lam * Tg)) / lam
    A = (th - eta**2 / (2 * lam**2)) * (B - Tg) - eta**2 * B**2 / (4 * lam)
    P_vas = np.exp(A - B * r0)

    def yld(P):
        return -np.log(P) / Tg

    fig = go.Figure()
    fig.add_scatter(x=Tg, y=yld(P_mkt), name="Market curve y(0, T)",
                    line=dict(color=NEUTRAL, width=6), opacity=0.5)
    fig.add_scatter(x=Tg, y=yld(P_mc), name="Hull-White Monte Carlo", mode="markers",
                    marker=dict(color=BLUE, size=9),
                    error_y=dict(type="data", array=1.96 * se / P_mc / Tg, color=BLUE))
    fig.add_scatter(x=Tg, y=yld(P_vas), name="Vasicek (constant θ̄ = β₀)",
                    line=dict(color=ORANGE, width=2, dash="dash"))
    fig.update_layout(xaxis_title="Maturity T (years)", yaxis_title="Zero rate",
                      yaxis_tickformat=".2%", height=450)
    st.plotly_chart(fig, width="stretch")
    err_hw = np.abs(yld(P_mc) - yld(P_mkt)).max() * 1e4
    err_vas = np.abs(yld(P_vas) - yld(P_mkt)).max() * 1e4
    c1, c2 = st.columns(2)
    c1.metric("Hull-White: largest yield error", f"{err_hw:.1f} bp")
    c2.metric("Vasicek: largest yield error", f"{err_vas:.1f} bp")
    st.caption(
        "Hull-White's error is just Monte Carlo noise (the error bars) plus a tiny Euler bias, "
        "for every curve shape you pick. Vasicek can only produce curves of one family. "
        "Try the 'Humped' preset. Fitting today's curve exactly matters: a model that "
        "misprices a plain zero-coupon bond can't be trusted on a swap or an option."
    )

# ---------------- Tab 4: future yield curves ----------------
with tab_future:
    st.markdown(
        "Hull-White gives the whole yield curve at a future time t in closed form, from the "
        "single number r(t) (book Lemma 11.3.1): P(t, T) = A(t, T)·exp(B(t, T)·r(t)). "
        "Below, five simulated scenarios at time t, from a low-rate path to a high-rate path."
    )
    c1, c2 = st.columns(2)
    t_f = c1.select_slider("Look at the curve at time t", [1.0, 2.0, 5.0], value=2.0)
    r_paths, _ = cached_paths(b0, b1, b2, tau, lam, eta, 10_000, 42)
    rt = r_paths[:, int(round(t_f / HORIZON * N_STEPS))]
    qs = [5, 25, 50, 75, 95]
    rq = np.percentile(rt, qs)
    taus = np.linspace(0.25, 15.0, 60)
    cols = [RAMP[0], RAMP[1], NEUTRAL, RAMP[2], RAMP[3]]

    fig = go.Figure()
    for q, rr, colour in zip(qs, rq, cols):
        y = [-np.log(hw.zcb(t_f, t_f + s, rr)) / s for s in taus]
        fig.add_scatter(x=taus, y=y, name=f"{q}th pct scenario: r({t_f:g}y) = {rr:.2%}",
                        line=dict(color=colour, width=2.5 if q == 50 else 2))
    fwd = [-(np.log(curve.discount(t_f + s)) - np.log(curve.discount(t_f))) / s for s in taus]
    fig.add_scatter(x=taus, y=fwd, name="Today's forward curve for time t",
                    line=dict(color=ORANGE, width=2, dash="dash"))
    fig.update_layout(xaxis_title="Remaining maturity T − t (years)",
                      yaxis_title="Zero rate at time t", yaxis_tickformat=".2%", height=460)
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for l_, colour in [(0.05, VIOLET), (lam, BLUE), (2.0, GREEN)]:
        fig.add_scatter(x=taus, y=(1 - np.exp(-l_ * taus)) / (l_ * taus),
                        name=f"λ = {l_:g}" + (" (your choice)" if l_ == lam else ""),
                        line=dict(color=colour, width=2.5 if l_ == lam else 1.5))
    fig.update_layout(xaxis_title="Remaining maturity T − t (years)",
                      yaxis_title="Yield move per 1% move in r", yaxis_tickformat=".0%",
                      height=330, yaxis_range=[0, 1.05])
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "One random number drives the whole curve, so every scenario moves the curve in the "
        "same pattern: a 1% rise in r lifts the T-year yield by (1 − e^{−λT})/(λT), which is "
        "close to 1 at the short end and fades with maturity. Large λ: long yields barely "
        "move. Small λ: near-parallel shifts. What a one-factor model can't produce is a twist, "
        "short rates up while long rates fall. That needs a second factor (book Ch 13, "
        "two-factor models). The median scenario lies almost on today's forward curve: the "
        "curve already contains the market's expectation of future rates; the small gap is "
        "the convexity term."
    )
