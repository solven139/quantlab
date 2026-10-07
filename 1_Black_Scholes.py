import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantlab import (
    OptionType,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
    norm_cdf,
    norm_pdf,
    simulate_gbm,
)

# Fixed series colours: calls are always blue, puts always orange, on every chart
CALL_COLOR = "#3987e5"
PUT_COLOR = "#d95926"
PATH_COLOR = "rgba(57, 135, 229, 0.35)"
REF_COLOR = "#c3c2b7"

st.title("Black-Scholes")
st.latex(r"dS(t) = r\,S(t)\,dt + \sigma\,S(t)\,dW^{\mathbb{Q}}(t)")
st.caption(
    "Assumptions: constant volatility, constant rate, no jumps, continuous hedging, "
    "no frictions. Book: Chapters 2-3."
)

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Market")
    S0 = st.slider("Spot S0", 50.0, 150.0, 100.0)
    K = st.slider("Strike K", 50.0, 150.0, 100.0)
    T = st.slider("Maturity T (years)", 0.05, 5.0, 1.0)
    r = st.slider("Rate r", 0.0, 0.10, 0.05, step=0.005)
    sigma = st.slider("Volatility sigma", 0.05, 0.80, 0.20, step=0.01)

# ---------------- Headline numbers ----------------
c = bs_price(OptionType.Call, S0, K, T, r, sigma)
p = bs_price(OptionType.Put, S0, K, T, r, sigma)
cols = st.columns(5)
cols[0].metric("Call price", f"{c:.4f}")
cols[1].metric("Put price", f"{p:.4f}")
cols[2].metric("Call delta", f"{bs_delta(OptionType.Call, S0, K, T, r, sigma):.4f}")
cols[3].metric("Gamma", f"{bs_gamma(S0, K, T, r, sigma):.4f}")
cols[4].metric("Vega", f"{bs_vega(S0, K, T, r, sigma):.4f}")

tab_greeks, tab_paths, tab_mc = st.tabs(
    ["Price & Greeks", "Simulated paths", "Monte Carlo vs formula"]
)

# ---------------- Tab 1: price and Greeks against spot ----------------
with tab_greeks:
    quantity = st.radio("Show", ["Price", "Delta", "Gamma", "Vega"], horizontal=True)
    spots = np.linspace(0.5 * K, 1.5 * K, 201)

    def curve(cp):
        if quantity == "Price":
            return [bs_price(cp, s, K, T, r, sigma) for s in spots]
        if quantity == "Delta":
            return [bs_delta(cp, s, K, T, r, sigma) for s in spots]
        if quantity == "Gamma":
            return [bs_gamma(s, K, T, r, sigma) for s in spots]
        return [bs_vega(s, K, T, r, sigma) for s in spots]

    fig = go.Figure()
    fig.add_scatter(x=spots, y=curve(OptionType.Call), name="Call",
                    line=dict(color=CALL_COLOR, width=2))
    if quantity in ("Price", "Delta"):
        fig.add_scatter(x=spots, y=curve(OptionType.Put), name="Put",
                        line=dict(color=PUT_COLOR, width=2))
    fig.add_vline(x=S0, line=dict(color=REF_COLOR, width=1, dash="dot"),
                  annotation_text="current S0")
    fig.update_layout(xaxis_title="Spot S0", yaxis_title=quantity, height=420,
                      hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    if quantity in ("Gamma", "Vega"):
        st.info(f"{quantity} is the same for calls and puts: put-call parity says "
                "C - P = S0 - K e^(-rT), whose second derivative in S0 and derivative "
                "in sigma are both zero.")

# ---------------- Tab 2: simulated GBM paths ----------------
# Blue ramp for "colour by final value": light = low S(T), dark = high S(T)
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#104281"]


@st.cache_data(show_spinner="Simulating paths in C++...")
def cached_paths(S0, r, sigma, T, n_steps, n_paths, seed):
    # Cached: moving a drawing-only control doesn't re-run the simulation
    return simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed)


def lines_trace(times, rows, color, name, show_legend):
    # Many paths in ONE trace, separated by NaN gaps: far faster than one trace per path
    n_t = len(times)
    x = np.tile(np.append(times, np.nan), len(rows))
    y = np.hstack([rows, np.full((len(rows), 1), np.nan)]).ravel()
    return go.Scattergl(x=x, y=y, mode="lines", name=name, showlegend=show_legend,
                        line=dict(color=color, width=1), opacity=0.55,
                        hoverinfo="skip", connectgaps=False)


with tab_paths:
    c1, c2, c3 = st.columns(3)
    n_sim = c1.select_slider("Paths simulated", [1_000, 5_000, 10_000, 25_000, 50_000],
                             value=10_000)
    n_draw = c2.slider("Paths drawn", 10, 1_000, 200, step=10)
    seed = c3.number_input("Random seed", value=42, step=1)
    view = st.radio("View", ["Individual paths", "Percentile fan (all simulated paths)"],
                    horizontal=True)
    colour_by = st.radio("Colour paths by", ["Above / below strike K at expiry",
                                              "Final value S(T)"], horizontal=True,
                         disabled=(view != "Individual paths"))

    n_steps = 100
    paths = cached_paths(S0, r, sigma, T, n_steps, n_sim, int(seed))
    times = np.linspace(0.0, T, n_steps + 1)
    ST = paths[:, -1]
    shown = paths[: min(n_draw, n_sim)]

    fig = go.Figure()
    if view == "Individual paths":
        if colour_by.startswith("Above"):
            above = shown[:, -1] > K
            fig.add_trace(lines_trace(times, shown[above], CALL_COLOR,
                                      f"Ends above K ({above.sum()}): call pays", True))
            fig.add_trace(lines_trace(times, shown[~above], PUT_COLOR,
                                      f"Ends below K ({(~above).sum()}): put pays", True))
            fig.add_hline(y=K, line=dict(color=REF_COLOR, width=1, dash="dot"),
                          annotation_text="strike K", annotation_position="bottom left")
        else:
            edges = np.quantile(shown[:, -1], np.linspace(0, 1, len(RAMP) + 1))
            bucket = np.clip(np.searchsorted(edges, shown[:, -1], side="right") - 1,
                             0, len(RAMP) - 1)
            for b, colour in enumerate(RAMP):
                fig.add_trace(lines_trace(times, shown[bucket == b], colour,
                                          f"S(T) {edges[b]:.0f}-{edges[b + 1]:.0f}", True))
    else:
        q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        for lo, hi, alpha, label in [(0, 4, 0.18, "5%-95% of paths"),
                                     (1, 3, 0.35, "25%-75% of paths")]:
            fig.add_scatter(x=np.concatenate([times, times[::-1]]),
                            y=np.concatenate([q[hi], q[lo][::-1]]),
                            fill="toself", fillcolor=f"rgba(57, 135, 229, {alpha})",
                            line=dict(width=0), name=label, hoverinfo="skip")
        fig.add_scatter(x=times, y=q[2], name="Median",
                        line=dict(color=CALL_COLOR, width=2, dash="dash"))

    # Scattergl so the mean line is drawn on the same WebGL layer as the paths, on top
    mean_colour = "#8a8985" if (view == "Individual paths"
                                and colour_by.startswith("Above")) else PUT_COLOR
    fig.add_trace(go.Scattergl(x=times, y=S0 * np.exp(r * times), mode="lines",
                               name="E[S(t)] = S0 e^(rt)",
                               line=dict(color=mean_colour, width=4)))
    fig.update_layout(xaxis_title="Time t (years)", yaxis_title="S(t)", height=460,
                      legend=dict(itemsizing="constant"))
    st.plotly_chart(fig, width="stretch")
    if view == "Individual paths" and colour_by.startswith("Above"):
        frac = (ST > K).mean()
        d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        st.caption(
            f"Of all {n_sim:,} simulated paths, {frac:.1%} end above K. Theory: the "
            f"risk-neutral probability that the call is exercised is N(d2) = "
            f"{norm_cdf(d2):.1%}. The {len(shown)} drawn paths are a smaller, noisier sample."
        )
    if view != "Individual paths":
        st.caption("The median sits below the mean: most paths end lower than average, "
                   "while a few large winners pull the mean up. That's the lognormal skew.")

    # Terminal distribution: histogram of all simulated S(T) vs the exact lognormal density
    grid = np.linspace(ST.min(), np.percentile(ST, 99.5), 300)
    sd = sigma * np.sqrt(T)
    density = [norm_pdf((np.log(s / S0) - (r - 0.5 * sigma**2) * T) / sd) / (s * sd)
               for s in grid]

    fig = go.Figure()
    fig.add_histogram(x=ST, histnorm="probability density", nbinsx=100,
                      name=f"Simulated S(T), {n_sim:,} paths",
                      marker=dict(color=CALL_COLOR, opacity=0.6))
    fig.add_scatter(x=grid, y=density, name="Lognormal density (exact)",
                    line=dict(color=PUT_COLOR, width=2))
    fig.update_layout(xaxis_title="S(T)", yaxis_title="Density", height=380,
                      bargap=0.02)
    st.plotly_chart(fig, width="stretch")
    stderr = ST.std(ddof=1) / np.sqrt(n_sim)
    st.caption(
        f"Sample mean of S(T): {ST.mean():.3f} ± {1.96 * stderr:.3f} (95%) vs theory "
        f"S0 e^(rT) = {S0 * np.exp(r * T):.3f}. More paths → narrower ± and a smoother "
        "histogram."
    )

# ---------------- Tab 3: Monte Carlo convergence ----------------
with tab_mc:
    st.markdown(
        "Price the call by averaging discounted payoffs over N simulated paths, "
        "and watch the estimate close in on the Black-Scholes formula. "
        "The band is the 95% confidence interval, which narrows like 1/√N."
    )
    ST_all = simulate_gbm(S0, r, sigma, T, 1, 200_000, int(seed))[:, -1]
    disc_payoff = np.exp(-r * T) * np.maximum(ST_all - K, 0.0)
    Ns = np.unique(np.logspace(2, np.log10(len(disc_payoff)), 40).astype(int))
    est = np.array([disc_payoff[:n].mean() for n in Ns])
    se = np.array([disc_payoff[:n].std(ddof=1) / np.sqrt(n) for n in Ns])

    fig = go.Figure()
    fig.add_scatter(x=np.concatenate([Ns, Ns[::-1]]),
                    y=np.concatenate([est + 1.96 * se, (est - 1.96 * se)[::-1]]),
                    fill="toself", fillcolor="rgba(57, 135, 229, 0.2)",
                    line=dict(width=0), name="95% confidence band", hoverinfo="skip")
    fig.add_scatter(x=Ns, y=est, name="Monte Carlo estimate",
                    line=dict(color=CALL_COLOR, width=2), mode="lines+markers",
                    marker=dict(size=8))
    fig.add_hline(y=c, line=dict(color=PUT_COLOR, width=2, dash="dash"),
                  annotation_text=f"Black-Scholes = {c:.4f}",
                  annotation_position="bottom right")
    fig.update_layout(xaxis_title="Number of paths N (log scale)",
                      yaxis_title="Call price", xaxis_type="log", height=420)
    st.plotly_chart(fig, width="stretch")
