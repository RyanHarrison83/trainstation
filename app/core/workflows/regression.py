import streamlit as st

def run(dataset_path, goal=None):
    """Placeholder regression workflow.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory.
    goal : str, optional
        Optimization goal selected by the user. Currently unused.
    """

    st.success("📊 Regression workflow started!")
    st.write("Launching FLAML or scikit-learn training.")
