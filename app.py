# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from functions import (
    input_preprocessing,
    preprocess_data,
    load_model_and_scaler,
    calculate_scores,
    predict_with_model,
    explain_emissions,
    generate_suggestions
)

# ---------- CONFIG ---------- #
st.set_page_config(page_title="Carbon Intelligence Platform", layout="wide")
sns.set_theme(style="whitegrid")

# ---------- CUSTOM CSS ---------- #
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- LOGO ---------- #
if os.path.exists("assets/logo.png"):
    st.sidebar.image("assets/logo.png", use_container_width=True)

st.sidebar.title("Carbon Intelligence Platform")

# ---------- BANNER ---------- #
if os.path.exists("assets/banner.png"):
    st.image("assets/banner.png", use_container_width=True)

st.title("Carbon Intelligence Platform")
st.caption("AI-powered Sustainability Analytics")

# ---------- NAV ---------- #
page = st.sidebar.radio("Navigation", ["Input", "Dashboard", "Insights"])

# ---------- INPUT ---------- #
if page == "Input":

    st.header("User Input")

    col1, col2 = st.columns(2)

    with col1:
        body_type = st.selectbox("Body Type", ["underweight", "normal", "overweight", "obese"])
        diet = st.selectbox("Diet", ["vegetarian", "non-vegetarian", "vegan"])
        transport = st.selectbox("Transport", ["walking", "bike", "public", "car"])
        energy_usage = st.slider("Energy Usage", 50, 500, 150)

    with col2:
        device_usage = st.slider("Device Usage", 1, 24, 5)
        internet_usage = st.slider("Internet Usage", 1, 50, 5)
        emails = st.number_input("Emails", 0, 500, 20)
        streaming_hours = st.slider("Streaming Hours", 0, 10, 2)
        video_calls = st.slider("Video Calls", 0, 10, 1)

    pue = st.slider("PUE", 1.0, 3.0, 1.5)
    carbon_intensity = st.slider("Carbon Intensity", 100, 900, 500)

    if st.button("Submit"):
        st.session_state["data"] = {
            "body_type": body_type,
            "diet": diet,
            "transport": transport,
            "energy_usage": energy_usage,
            "device_usage": device_usage,
            "internet_usage": internet_usage,
            "emails": emails,
            "streaming_hours": streaming_hours,
            "video_calls": video_calls,
            "pue": pue,
            "carbon_intensity": carbon_intensity
        }
        st.success("Data stored successfully!")

# ---------- DASHBOARD ---------- #
elif page == "Dashboard":

    if "data" not in st.session_state:
        st.warning("Please enter data first.")
    else:
        df = pd.DataFrame([st.session_state["data"]])

        lifestyle = input_preprocessing(df.copy())
        digital = preprocess_data(df.copy())

        model, scaler = load_model_and_scaler()

        lifestyle_score, digital_score = calculate_scores(lifestyle, digital)
        total = predict_with_model(model, scaler, df, lifestyle_score, digital_score)

        st.header("Carbon Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Lifestyle Score", round(lifestyle_score, 2))
        col2.metric("Digital Score", round(digital_score, 2))
        col3.metric("Total Emission", round(total, 2))

        # Chart 1
        fig, ax = plt.subplots()
        sns.barplot(x=["Lifestyle", "Digital"], y=[lifestyle_score, digital_score], ax=ax)
        st.pyplot(fig)

        # Chart 2
        components = explain_emissions(digital)

        fig2, ax2 = plt.subplots()
        sns.barplot(x=list(components.keys()), y=list(components.values()), ax=ax2)
        plt.xticks(rotation=20)
        st.pyplot(fig2)

# ---------- INSIGHTS ---------- #
elif page == "Insights":

    if "data" not in st.session_state:
        st.warning("Please enter data first.")
    else:
        df = pd.DataFrame([st.session_state["data"]])
        digital = preprocess_data(df)

        st.header("AI Insights")

        explain = explain_emissions(digital)

        for k, v in explain.items():
            st.write(f"{k}: {round(v, 2)}")

        st.subheader("Recommendations")

        for s in generate_suggestions(digital, df):
            st.write("- " + s)