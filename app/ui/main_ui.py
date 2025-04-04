import streamlit as st
import os
from core.workflow_manager import run_task

def launch_ui():
    st.set_page_config(page_title="TrainStation", layout="centered")
    st.title("🚉 TrainStation")
    st.caption("All aboard for fast, guided ML experimentation.")

    st.markdown("### 🧠 Select your ML Task")
    task = st.selectbox("Choose a task to begin:", [
        "Image Classification",
        "Object Detection",
        "Time Series Forecasting",
        "Regression"
    ])

    if task:
        st.markdown("### 🗂️ Select Your Dataset")
        dataset_path = st.text_input(
            "Enter the path to your dataset folder (e.g. with class subfolders):", 
            placeholder="/workspace/data/my_dataset"
        )

        if os.path.exists(dataset_path):
            st.success("✅ Valid dataset path found.")

            # Optional goal selector (only for certain tasks)
            goal = None
            if task in ["Image Classification", "Regression"]:
                st.markdown("### 🎯 Choose Optimization Goal")
                goal = st.selectbox("What do you want to prioritize?", [
                    "Highest Accuracy",
                    "Fastest Training",
                    "Best Generalization"
                ])

            st.markdown("### 🏁 Ready to Train")
            if st.button("Train Model"):
                with st.spinner("🚂 Training in progress... Please wait..."):
                    result = run_task(task, dataset_path, goal)

                st.success("✅ Training complete!")

                
        else:
            st.warning("🚫 Please enter a valid dataset folder path.")
