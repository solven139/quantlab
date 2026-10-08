import streamlit as st

REPO_URL = "https://github.com/solven139/quantlab"   # change if your repo has another name

st.set_page_config(page_title="QuantLab", page_icon="📈", layout="wide")

st.title("QuantLab")
st.markdown(
    "#### An interactive tour of the models in *Mathematical Modeling and Computation in "
    "Finance* (Oosterlee & Grzelak)"
)
st.markdown(
    "Every model in quantitative finance rests on assumptions: constant volatility, no "
    "jumps, a fixed interest rate, a counterparty that always pays. Each page here starts "
    "from one model, lets you change its parameters, and shows what happens when an "
    "assumption is relaxed. You can watch the paths, the smiles and the exposure profiles "
    "respond, instead of only reading the formulas."
)
st.caption(
    "All pricing and simulation runs in a C++20 engine (Black-Scholes, COS method, jump "
    "models, Heston, Monte Carlo schemes, Dupire local volatility, Hull-White, exposure and "
    "CVA), exposed to Python with pybind11 and checked against the book's reference values "
    "by an automated test suite."
)

SECTIONS = [
    ("Black-Scholes world", [
        ("pages/1_Black_Scholes.py", "Black-Scholes", "📐",
         "Prices, Greeks, GBM paths and Monte Carlo vs the formula.", "Ch 2–3"),
        ("pages/2_Delta_Hedging.py", "Delta Hedging", "🛡️",
         "Replicate an option and see the hedging error shrink with rebalancing.", "Ch 3"),
        ("pages/3_Implied_Volatility.py", "Implied Volatility", "🔍",
         "Newton vs bisection, and the smile the market quotes.", "Ch 4.1"),
    ]),
    ("Beyond constant volatility", [
        ("pages/4_COS_Method.py", "COS Method", "〰️",
         "Price from the characteristic function with exponential convergence.", "Ch 6"),
        ("pages/5_Jump_Models.py", "Jump Models", "⚡",
         "Merton, Kou and Variance Gamma: fat tails and short-dated skew.", "Ch 5"),
        ("pages/6_Heston.py", "Heston", "🌊",
         "Stochastic volatility: what κ, γ, ρ and v₀ do to the smile.", "Ch 8"),
        ("pages/7_Monte_Carlo_Lab.py", "Monte Carlo Lab", "🎲",
         "Euler vs Milstein, exact CIR, the QE scheme and discretisation bias.", "Ch 9"),
        ("pages/8_Local_Volatility.py", "Local Volatility", "🗺️",
         "Dupire's surface: same prices today, different future smiles.", "Ch 4.3, 10"),
    ]),
    ("Interest rates and counterparty risk", [
        ("pages/9_Hull_White.py", "Hull-White", "📉",
         "A short rate fitted to today's curve, and the curves it generates.", "Ch 11"),
        ("pages/10_Exposure_and_CVA.py", "Exposure and CVA", "🏦",
         "Swap exposure, EE and PFE, CVA, DVA and netting.", "Ch 12"),
    ]),
]

for title, pages in SECTIONS:
    st.subheader(title)
    cols = st.columns(3)
    for i, (path, label, icon, blurb, ch) in enumerate(pages):
        with cols[i % 3].container(border=True):
            st.page_link(path, label=f"**{label}**", icon=icon)
            st.caption(f"{blurb}  \nBook {ch}")

st.divider()
st.caption(
    f"Built as a learning project by Scott-C (MSc Quantitative Finance, SMU). Source code, "
    f"tests and build instructions: [GitHub]({REPO_URL})."
)
