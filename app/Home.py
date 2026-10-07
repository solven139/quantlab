import streamlit as st

st.set_page_config(page_title="QuantLab", layout="wide")
st.title("QuantLab")
st.markdown(
    """
An interactive tour of the models in *Mathematical Modeling and Computation in Finance*
(Oosterlee & Grzelak). Each page starts from one model, shows the assumptions it makes,
and lets you see what changes when an assumption is relaxed.

Pick a model from the sidebar.
"""
)
