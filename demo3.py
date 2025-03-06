import streamlit as st
import pandas as pd
import datetime
import time
import requests
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def get_location_coordinates(location_name):
    geolocator = Nominatim(user_agent="disaster_predictor", timeout=10)
    time.sleep(2)
    location = geolocator.geocode(location_name)
    return (location.latitude, location.longitude) if location else (None, None)

def fetch_earthquake_data(location_name):
    lat, lon = get_location_coordinates(location_name)
    if lat is None or lon is None:
        st.error("Unable to fetch coordinates for the location.")
        return pd.DataFrame()
    
    url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
           f"format=geojson&latitude={lat}&longitude={lon}&maxradiuskm=500&starttime=2000-01-01")
    try:
        time.sleep(2)
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if not data['features']:
                st.warning("No earthquake data found for this location.")
                return pd.DataFrame()
            earthquake_list = [{
                "Date": datetime.datetime.fromtimestamp(q['properties']['time'] / 1000),
                "Magnitude": q['properties']['mag'],
                "Place": q['properties']['place'],
                "Longitude": q['geometry']['coordinates'][0],
                "Latitude": q['geometry']['coordinates'][1],
                "Severity": classify_earthquake(q['properties']['mag'])
            } for q in data['features']]
            return pd.DataFrame(earthquake_list)
        else:
            st.error(f"API request failed with status code {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching earthquake data: {e}")
        return pd.DataFrame()

def classify_earthquake(magnitude):
    if magnitude < 4.0:
        return "Minor"
    elif magnitude < 6.0:
        return "Moderate"
    elif magnitude < 7.0:
        return "Strong"
    elif magnitude < 8.0:
        return "Major"
    else:
        return "Great"

def fetch_flood_data():
    flood_data = {
        "Date": ["2010-07-20", "2011-08-15", "2013-09-12", "2016-06-05", "2017-07-10", "2018-09-25", "2020-08-20", "2023-07-18"],
        "Flood_Severity": ["Low", "Medium", "High", "High", "Low", "Medium", "High", "Very High"],
        "Affected_Area": ["Delhi", "Delhi", "Delhi", "Delhi", "Delhi", "Delhi", "Delhi", "Delhi"]
    }
    return pd.DataFrame(flood_data)

def train_predict_disaster_model(earthquake_df, flood_df):
    earthquake_df["Disaster"] = "Earthquake"
    flood_df["Disaster"] = "Flood"
    flood_df["Date"] = pd.to_datetime(flood_df["Date"])
    df = pd.concat([earthquake_df, flood_df], ignore_index=True)
    df["Timestamp"] = df["Date"].astype('int64') // 10**9
    encoder = LabelEncoder()
    df["Disaster_Type"] = encoder.fit_transform(df["Disaster"])
    X = df[["Timestamp"]]
    y = df["Disaster_Type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    future_timestamp = int(datetime.datetime.now().timestamp()) + (30 * 24 * 60 * 60)
    future_prediction = model.predict(pd.DataFrame([[future_timestamp]], columns=["Timestamp"]))
    predicted_disaster = encoder.inverse_transform(future_prediction)[0]
    predicted_severity = "Unknown"
    if predicted_disaster == "Earthquake" and not earthquake_df.empty:
        predicted_severity = earthquake_df["Severity"].value_counts().idxmax()
    elif predicted_disaster == "Flood" and not flood_df.empty:
        predicted_severity = flood_df["Flood_Severity"].value_counts().idxmax()
    return predicted_disaster, predicted_severity

st.title("🌍 Disaster Prediction System")
city = st.text_input("Enter city name (e.g., New Delhi)")
if st.button("Predict Disaster"):
    with st.spinner("Fetching earthquake data..."):
        earthquake_data = fetch_earthquake_data(city)
    if not earthquake_data.empty:
        st.success("Earthquake data retrieved successfully!")
        st.dataframe(earthquake_data.sort_values(by="Date", ascending=False).head(5))
    with st.spinner("Fetching flood data..."):
        flood_data = fetch_flood_data()
    st.success("Flood data retrieved successfully!")
    st.dataframe(flood_data.sort_values(by="Date", ascending=False).head(5))
    with st.spinner("Training prediction model..."):
        predicted_disaster, predicted_severity = train_predict_disaster_model(earthquake_data, flood_data)
    st.subheader(f"🔮 Prediction: The next disaster in {city} is likely to be a {predicted_disaster} ({predicted_severity} severity).")
