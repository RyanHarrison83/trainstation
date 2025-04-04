import streamlit as st
import autokeras as ak
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import os

def run(dataset_path, goal=None):
    st.success("🧪 Starting AutoKeras Image Classification")
    st.write(f"📁 Using dataset from: `{dataset_path}`")
    if goal:
        st.write(f"🎯 Optimization goal: `{goal}`")

    try:
        # Load training dataset
        train_dataset = image_dataset_from_directory(
            dataset_path,
            image_size=(224, 224),
            batch_size=32,
            validation_split=0.2,
            subset="training",
            seed=123,
        )
        val_dataset = image_dataset_from_directory(
            dataset_path,
            image_size=(224, 224),
            batch_size=32,
            validation_split=0.2,
            subset="validation",
            seed=123,
        )
        st.success("✅ Dataset loaded successfully")
    except Exception as e:  
        st.error(f"❌ Error loading dataset: {e}")
        return
    
    # Display a few images from the dataset
    st.subheader("Sample Images from Dataset")
    sample_images = []
    sample_labels = []
    class_names = train_dataset.class_names
    for images, labels in train_dataset.take(1):
        sample_images = images.numpy()
        sample_labels = labels.numpy()
        break
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(sample_images[i].astype("uint8"))
        axes[i].set_title(class_names[sample_labels[i]])
        axes[i].axis("off")
    st.pyplot(fig)
   
    # Initialize AutoKeras ImageClassifier
    st.subheader("Initializing AutoKeras Image Classifier")
    model = ak.ImageClassifier(max_trials=10, overwrite=True)
    st.success("✅ AutoKeras Image Classifier initialized")
    st.write("🔄 Training the model...")
    # start a loading spinner
    with st.spinner("Training in progress..."):
        pass
        # Train the model
        try:
            model.fit(train_dataset, epochs=10, validation_data=val_dataset)
            st.success("✅ Model training completed")
        except Exception as e:
            st.error(f"❌ Error during model training: {e}")
            return

    