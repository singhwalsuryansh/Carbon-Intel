# functions.py

import pickle
import os

def input_preprocessing(df):
    df["body_type"] = df["body_type"].map({"underweight":0,"normal":1,"overweight":2,"obese":3})
    df["diet"] = df["diet"].map({"vegetarian":0,"non-vegetarian":1,"vegan":2})
    df["transport"] = df["transport"].map({"walking":0,"bike":1,"public":2,"car":3})
    return df

def preprocess_data(df):
    df["email_emission"] = df["emails"] * 4
    df["streaming_emission"] = df["streaming_hours"] * 36
    df["video_emission"] = df["video_calls"] * 150
    df["device_emission"] = df["device_usage"] * 20
    df["internet_emission"] = df["internet_usage"] * 5
    df["infra_factor"] = df["pue"] * (df["carbon_intensity"]/500)
    return df

def load_model_and_scaler():
    model, scaler = None, None
    if os.path.exists("model.pkl"):
        model = pickle.load(open("model.pkl","rb"))
    if os.path.exists("scaler.pkl"):
        scaler = pickle.load(open("scaler.pkl","rb"))
    return model, scaler

def calculate_scores(lifestyle, digital):
    lifestyle_score = float(
        lifestyle["body_type"][0]*10 +
        lifestyle["diet"][0]*20 +
        lifestyle["transport"][0]*30 +
        lifestyle["energy_usage"][0]*0.5
    )

    digital_score = float((
        digital["email_emission"][0] +
        digital["streaming_emission"][0] +
        digital["video_emission"][0] +
        digital["device_emission"][0] +
        digital["internet_emission"][0]
    ) * digital["infra_factor"][0])

    return lifestyle_score, digital_score

def predict_with_model(model, scaler, df, lifestyle, digital):
    if model and scaler:
        try:
            X = df.copy()
            X["body_type"] = X["body_type"].map({"underweight":0,"normal":1,"overweight":2,"obese":3})
            X["diet"] = X["diet"].map({"vegetarian":0,"non-vegetarian":1,"vegan":2})
            X["transport"] = X["transport"].map({"walking":0,"bike":1,"public":2,"car":3})

            X_scaled = scaler.transform(X)
            return float(model.predict(X_scaled)[0])
        except:
            pass

    return lifestyle + digital

def explain_emissions(data):
    return {
        "Emails": data["email_emission"][0],
        "Streaming": data["streaming_emission"][0],
        "Video Calls": data["video_emission"][0],
        "Devices": data["device_emission"][0],
        "Internet": data["internet_emission"][0]
    }

def generate_suggestions(data, raw):
    contributions = explain_emissions(data)
    max_source = max(contributions, key=contributions.get)

    suggestions = [f"Highest emission source: {max_source}"]

    if max_source == "Streaming":
        suggestions.append("Reduce streaming quality")
    elif max_source == "Video Calls":
        suggestions.append("Use audio calls instead")
    elif max_source == "Emails":
        suggestions.append("Limit unnecessary emails")
    elif max_source == "Devices":
        suggestions.append("Enable power saving mode")

    if raw["transport"][0] == "car":
        suggestions.append("Switch to public transport")

    return suggestions