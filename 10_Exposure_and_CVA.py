import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from quantlab import (HullWhiteModel, NelsonSiegelCurve, Swap, cva, par_swap_rate,
                      simulate_exposure, swap_annuity, swap_value_today)

BLUE = "#3987e5"
ORANGE = "#d95926"
GREEN = "#1baf7a"
VIOLET = "#4a3aa7"
NEUTRAL = "#8a8985"
PATH_BLUE = "rgba(57, 135, 229, 0.30)"
PATH_ORANGE = "rgba(217, 89, 38, 0.30)"

st.title("Exposure and CVA")
st.markdown(
    "Every price so far assumed the other side of the trade always pays. In 2008 that "
    "assumption failed. **Counterparty credit risk** is the loss you take if your "
    "counterparty defaults while the trade is worth something to you (book Ch 12.3). "
    "Its price is the **credit valuation adjustment**:"
)
st.latex(r"\text{risky value} = \text{risk-free value} - \text{CVA}")
st.latex(r"\text{CVA} \approx \underbrace{(1-R)}_{\text{LGD}}\sum_k "
         r"\underbrace{\mathrm{EE}(t_k)}_{\text{expected exposure}}\;"
         r"\underbrace{\big(e^{-h t_{k-1}} - e^{-h t_k}\big)}_{\text{prob. of default}}")
st.caption(
    "The trade here is an interest rate swap. Its value today comes from the yield curve "
    "alone, but its value tomorrow depends on where rates go, so we simulate them with the "
    "Hull-White model from the previous page and revalue the swap on every path."
)

PRESETS = {
    "Upward sloping (normal)": (0.04, -0.02, 0.01, 2.0),
    "Inverted": (0.03, 0.02, -0.01, 1.5),
    "Flat": (0.03, 0.0, 0.0, 2.0),
}

with st.sidebar:
    st.header("Swap")
    notional = 10_000.0
    maturity = st.select_slider("Maturity (years)", [2, 5, 10, 15], value=10)
    freq = st.radio("Payments", ["Annual", "Semi-annual", "Quarterly"], index=1, horizontal=True)
    tau = {"Annual": 1.0, "Semi-annual": 0.5, "Quarterly": 0.25}[freq]
    side = st.radio("Position", ["Payer (pay fixed)", "Receiver (receive fixed)"])
    payer = side.startswith("Payer")
    k_shift = st.slider("Fixed rate K vs par (bp)", -150, 150, 0, step=10)
    st.caption(f"Notional N = {notional:,.0f}")
    st.header("Rates (Hull-White)")
    preset = st.selectbox("Today's curve", list(PRESETS))
    b0, b1, b2, ns_tau = PRESETS[preset]
    lam = st.slider("λ  mean reversion", 0.02, 1.5, 0.3, step=0.02)
    eta = st.slider("η  rate volatility", 0.002, 0.03, 0.01, step=0.001, format="%.3f")
    n_paths = st.select_slider("Monte Carlo paths", [1_000, 2_000, 5_000, 10_000], value=5_000)

curve = NelsonSiegelCurve(b0=b0, b1=b1, b2=b2, tau=ns_tau)
hw = HullWhiteModel(lambda_=lam, eta=eta, curve=curve)
swap = Swap(notional=notional, K=0.0, start=0.0, end=float(maturity), tau=tau, payer=payer)
par = par_swap_rate(curve, swap)
swap.K = par + k_shift / 1e4
v0 = swap_value_today(curve, swap)
annuity = swap_annuity(curve, swap)


def make_swap(K, end, tau, payer):
    return Swap(notional=notional, K=K, start=0.0, end=float(end), tau=tau, payer=payer)


@st.cache_data(show_spinner="Simulating rates and revaluing the swaps on every path (C++)...")
def run(curve_p, lam, eta, trades_p, n_paths):
    m = HullWhiteModel(lambda_=lam, eta=eta, curve=NelsonSiegelCurve(*curve_p))
    trades = [Swap(notional=n, K=K, start=0.0, end=e, tau=t, payer=p) for n, K, e, t, p in trades_p]
    horizon = max(t[2] for t in trades_p)
    return simulate_exposure(m, trades, horizon, 24, n_paths, 11)


def profiles(T, V, D):
    E = np.maximum(V, 0.0)
    return dict(
        EE=(D * E).mean(0),
        ENE=(D * np.maximum(-V, 0.0)).mean(0),
        PFE95=np.percentile(E, 95, axis=0),
        PFE99=np.percentile(E, 99, axis=0),
    )


def lines_trace(x_times, rows, color, name, showlegend=True):
    x = np.tile(np.append(x_times, np.nan), len(rows))
    y = np.hstack([rows, np.full((len(rows), 1), np.nan)]).ravel()
    return go.Scattergl(x=x, y=y, mode="lines", name=name, showlegend=showlegend,
                        line=dict(color=color, width=1), hoverinfo="skip", connectgaps=False)


trade1 = (notional, swap.K, float(maturity), tau, payer)
T, V, D = run((b0, b1, b2, ns_tau), lam, eta, (trade1,), n_paths)
prof = profiles(T, V[0], D)

tab_swap, tab_paths, tab_ee, tab_cva, tab_net = st.tabs(
    ["The swap", "Exposure paths", "EE and PFE", "CVA", "Netting"]
)

# ---------------- Tab 1: the swap ----------------
with tab_swap:
    c1, c2, c3 = st.columns(3)
    c1.metric("Par swap rate", f"{par:.3%}", help="Book eq. 12.14: the K with value zero today")
    c2.metric(f"Fixed rate K ({k_shift:+d} bp vs par)", f"{swap.K:.3%}")
    c3.metric("Value today V(0)", f"{v0:,.1f}", help="Book eq. 12.12, from the curve alone")

    dates = np.array(swap.payment_dates())
    prev = np.concatenate([[0.0], dates[:-1]])
    P = np.array([curve.discount(t) for t in dates])
    Pprev = np.array([curve.discount(t) for t in prev])
    libor = (Pprev / P - 1.0) / tau            # today's forward Libor for each period
    sign = 1.0 if payer else -1.0
    fig = go.Figure()
    fig.add_bar(x=dates, y=sign * notional * tau * libor, name="Floating coupons N·τ·ℓ (expected)",
                marker_color=BLUE)
    fig.add_bar(x=dates, y=-sign * notional * tau * swap.K * np.ones_like(dates),
                name="Fixed coupons N·τ·K", marker_color=ORANGE)
    fig.add_scatter(x=dates, y=sign * notional * tau * (libor - swap.K), mode="lines+markers",
                    name="Net cash flow", line=dict(color=NEUTRAL, width=2))
    fig.update_layout(barmode="relative", xaxis_title="Payment date T_k (years)",
                      yaxis_title="Cash flow (+ received, − paid)", height=400)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "With an upward-sloping curve the expected floating coupons start below the par rate "
        "and end above it. The payer loses money in the early periods and makes it back later, "
        "and at par the two exactly balance in present value. That timing is why exposure "
        "builds up over the life of the swap: in the later years the payer is owed money. "
        "Floating coupons shown are today's forward rates, the expected values; on any one "
        "path they come out higher or lower."
    )

# ---------------- Tab 2: paths (book Fig 12.5) ----------------
with tab_paths:
    n_draw = st.slider("Paths drawn", 20, 400, 100, step=20)
    rows = V[0][:n_draw]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("Swap value V(t)", "Exposure E(t) = max(V(t), 0)"))
    pos_end = rows[:, len(T) // 2] > 0
    fig.add_trace(lines_trace(T, rows[pos_end], PATH_BLUE, "V > 0 at mid-life"), 1, 1)
    fig.add_trace(lines_trace(T, rows[~pos_end], PATH_ORANGE, "V < 0 at mid-life"), 1, 1)
    fig.add_trace(lines_trace(T, np.maximum(rows[pos_end], 0), PATH_BLUE, "", False), 1, 2)
    fig.add_trace(lines_trace(T, np.maximum(rows[~pos_end], 0), PATH_ORANGE, "", False), 1, 2)
    fig.update_xaxes(title_text="Time t (years)")
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_layout(height=450, legend=dict(itemsizing="constant"))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Every path starts at today's value and ends at 0 after the last payment. In between, "
        "rates wander and the swap drifts into the money for one side. Your loss on default "
        "is only the positive part: if you owe the defaulter, you still have to pay them "
        "(book Ch 12.3.1). That's why exposure, not value, drives CVA."
    )

# ---------------- Tab 3: EE and PFE (book Fig 12.6) ----------------
with tab_ee:
    fig = go.Figure()
    fig.add_scatter(x=T, y=prof["PFE99"], name="PFE 99%", line=dict(color=VIOLET, width=1.5, dash="dot"))
    fig.add_scatter(x=T, y=prof["PFE95"], name="PFE 95%", line=dict(color=VIOLET, width=2))
    fig.add_scatter(x=T, y=prof["EE"], name="Expected exposure EE(t)", line=dict(color=BLUE, width=3))
    fig.add_scatter(x=T, y=prof["ENE"], name="Expected negative exposure ENE(t)",
                    line=dict(color=ORANGE, width=2, dash="dash"))
    fig.update_layout(xaxis_title="Time t (years)", yaxis_title="Exposure", height=460)
    st.plotly_chart(fig, width="stretch")
    i_pk = int(np.argmax(prof["EE"]))
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Peak EE (at t = {T[i_pk]:.1f}y)", f"{prof['EE'][i_pk]:,.1f}")
    c2.metric("Peak PFE 95%", f"{prof['PFE95'].max():,.1f}")
    c3.metric("Peak EE / notional", f"{prof['EE'][i_pk] / notional:.2%}")
    st.caption(
        "Two forces shape the hump. Uncertainty grows with time, so the swap can drift further "
        "from 0, but every payment removes a coupon, so less is left to lose. Exposure peaks "
        "a quarter to a third of the way through, and the sawtooth comes from the payment dates "
        "(book Fig 12.6). ENE is your counterparty's exposure to you. Switch between payer and "
        "receiver: on an upward curve the payer's EE is larger, because the swap is expected "
        "to move in its favour. Higher η scales everything up; higher λ damps long-run rate "
        "moves and shrinks it."
    )

# ---------------- Tab 4: CVA ----------------
with tab_cva:
    c1, c2, c3 = st.columns(3)
    R = c1.slider("Recovery rate R", 0.0, 0.9, 0.4, step=0.05)
    spread = c2.slider("Counterparty credit spread (bp)", 0, 1000, 200, step=10)
    own = c3.slider("Your own credit spread (bp)", 0, 1000, 100, step=10)
    lgd = 1.0 - R
    h = spread / 1e4 / max(lgd, 1e-9)        # credit triangle: spread ≈ h · LGD
    h_own = own / 1e4 / max(lgd, 1e-9)
    tl = list(T)
    cva_v = cva(tl, list(prof["EE"]), lgd, h)
    dva_v = cva(tl, list(prof["ENE"]), lgd, h_own)

    # Contribution of each year to CVA
    yrs = np.arange(1, maturity + 1)
    # same sum as the C++ cva(), split by year: LGD * EE(t_k) * PD(t_{k-1}, t_k)
    step = lgd * prof["EE"][1:] * (np.exp(-h * T[:-1]) - np.exp(-h * T[1:]))
    year = np.minimum(np.ceil(T[1:] - 1e-9).astype(int), maturity) - 1
    contrib = np.zeros(maturity)
    np.add.at(contrib, year, step)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=yrs - 0.5, y=contrib, name="CVA contribution per year", marker_color=BLUE,
                width=0.8)
    fig.add_scatter(x=T, y=1 - np.exp(-h * T), name="Prob. of default by t", secondary_y=True,
                    line=dict(color=ORANGE, width=2))
    fig.update_xaxes(title_text="Time (years)")
    fig.update_yaxes(title_text="CVA contribution", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative default probability", tickformat=".0%",
                     rangemode="tozero", secondary_y=True)
    fig.update_yaxes(rangemode="tozero", secondary_y=False)
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk-free value", f"{v0:,.1f}")
    c2.metric(f"CVA (= {cva_v / (annuity * notional) * 1e4:.1f} bp running)", f"{cva_v:,.1f}")
    c3.metric("DVA", f"{dva_v:,.1f}", help="Your own default risk, seen by the counterparty")
    c4.metric("Bilateral value V − CVA + DVA", f"{v0 - cva_v + dva_v:,.1f}")
    st.caption(
        f"Hazard rate h = spread / LGD = {h:.2%} per year, so the counterparty defaults within "
        f"{maturity} years with probability {1 - np.exp(-h * maturity):.1%}. CVA weighs the EE "
        "curve by when default is likely, which is why the bars follow the exposure hump. "
        "'bp running' converts CVA into an extra fixed rate on the swap: the amount a bank "
        "would add to K for this counterparty. DVA is the mirror image, the counterparty's CVA "
        "to you, so in the bilateral price your own riskiness raises the value, the "
        "uncomfortable 'profit from your own default' (book Ch 12.3.3). This assumes default "
        "is independent of rates; when they're linked badly, that's wrong-way risk."
    )

# ---------------- Tab 5: netting (book Ch 12.3.4) ----------------
with tab_net:
    st.markdown(
        "With several trades against the same counterparty, a **netting agreement** lets you "
        "offset them on default: you lose max(V₁ + V₂, 0) instead of max(V₁, 0) + max(V₂, 0)."
    )
    c1, c2, c3 = st.columns(3)
    m2 = c1.select_slider("Second trade: maturity", [2, 5, 10, 15],
                          value=5 if maturity >= 5 else maturity)
    side2 = c2.radio("Second trade: position", ["Opposite of trade 1", "Same as trade 1"])
    k2 = c3.slider("Second trade: K vs its par (bp)", -150, 150, 0, step=10)
    payer2 = (not payer) if side2.startswith("Opposite") else payer
    s2 = make_swap(0.0, m2, tau, payer2)
    s2.K = par_swap_rate(curve, s2) + k2 / 1e4
    trade2 = (notional, s2.K, float(m2), tau, payer2)
    T2, V2, D2 = run((b0, b1, b2, ns_tau), lam, eta, (trade1, trade2), n_paths)

    ee_sep = (D2 * (np.maximum(V2[0], 0) + np.maximum(V2[1], 0))).mean(0)
    ee_net = (D2 * np.maximum(V2[0] + V2[1], 0)).mean(0)
    fig = go.Figure()
    fig.add_scatter(x=T2, y=ee_sep, name="EE without netting: E[max(V₁,0) + max(V₂,0)]",
                    line=dict(color=ORANGE, width=2.5))
    fig.add_scatter(x=T2, y=ee_net, name="EE with netting: E[max(V₁ + V₂, 0)]",
                    line=dict(color=BLUE, width=2.5))
    fig.update_layout(xaxis_title="Time t (years)", yaxis_title="Expected exposure", height=420)
    st.plotly_chart(fig, width="stretch")

    R_n, h_n = 0.4, 0.02 / 0.6
    cva_sep = cva(list(T2), list(ee_sep), 0.6, h_n)
    cva_net = cva(list(T2), list(ee_net), 0.6, h_n)
    c1, c2, c3 = st.columns(3)
    c1.metric("CVA without netting", f"{cva_sep:,.1f}")
    c2.metric("CVA with netting", f"{cva_net:,.1f}")
    c3.metric("Saving", f"{1 - cva_net / max(cva_sep, 1e-12):.0%}")
    st.caption(
        "Because max(x + y, 0) ≤ max(x, 0) + max(y, 0), netting can only help. Trades in "
        "opposite directions offset the most: one gains when the other loses. Two trades in "
        "the same direction barely net at all. (CVA here uses R = 40% and a 200 bp spread.)"
    )
    st.markdown("**Book Table 12.1**: deal 1 is worth +100, deal 2 −50, recovery 40%.")
    st.table({
        "": ["Deal 1", "Deal 2", "Total after default"],
        "Without netting": ["recover 40% × 100 = 40", "pay −50 in full", "−10"],
        "With netting": ["+100", "−50", "recover 40% × 50 = +20"],
    })
