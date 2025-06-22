import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown

# ------------------------ Download files from Google Drive ------------------------

file_ids = {
    'race_model.pkl': '1bUyue5WOCeShzWaUt8EtSgr7Juh32RGA',
    'scaler.pkl': '1pa9mV7jGBmea7yEBhMy-jMahqCLY7jO1',
    'config_encoder.pkl': '1AFmVVjSaZCebas6KM3FgG_gAfXOjnZ83',
    'going_encoder.pkl': '1qLWOB39AHiwNxNQe8QDj-RwCckxpD16m',
    'venue_encoder.pkl': '1FMpg8ulform9HDiVfwmWNE5bnSYqdLli',
    'horse_country_encoder.pkl': '1uSnj2xV0Kk5c27HMOJ2nE0_d0zO9bc77',
    'horse_type_encoder.pkl': '1OVS7_nif40xzku9RXjejwCT5ac4FuSri',
    'surface_encoder.pkl': '1uliRimvoQyYvSvltVAsAu65iUSaDZI4y',
}

os.makedirs("models", exist_ok=True)

for filename, file_id in file_ids.items():
    file_path = f"models/{filename}"
    if not os.path.exists(file_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)

# ------------------------ Load models and encoders ------------------------

model = joblib.load("models/race_model.pkl")
scaler = joblib.load("models/scaler.pkl")
config_encoder = joblib.load("models/config_encoder.pkl")
going_encoder = joblib.load("models/going_encoder.pkl")
venue_encoder = joblib.load("models/venue_encoder.pkl")
horse_country_encoder = joblib.load("models/horse_country_encoder.pkl")
horse_type_encoder = joblib.load("models/horse_type_encoder.pkl")
surface_encoder = joblib.load("models/surface_encoder.pkl")

# ------------------------ Streamlit UI ------------------------

st.title("🏇 Horse Race Finish Time Predictor - Multiple Horses")

st.write("Enter details for each horse to predict and compare their expected finish times:")

num_horses = st.number_input("Number of Horses", min_value=2, max_value=10, value=3, step=1)

horses_data = []
horse_names = []

for i in range(num_horses):
    st.markdown(f"### Horse {i + 1}")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            horse_name = st.text_input("Horse Name", value=f"Horse {i+1}", key=f"name_{i}")
            horse_age = st.number_input("Age", min_value=2, max_value=10, value=3, key=f"age_{i}")
            actual_wt = st.number_input("Actual Weight (kg)", min_value=100.0, max_value=150.0, value=120.0, key=f"awt_{i}")
            declared_wt = st.number_input("Declared Weight (kg)", min_value=600.0, max_value=1400.0, value=800.0, key=f"dwt_{i}")

        with col2:
            horse_country = st.selectbox("Country", [
                'ARG', 'AUS', 'BRZ', 'CAN', 'FR', 'GB', 'GER', 'GR', 'IRE', 'ITY', 'JPN', 'NZ', 'SAF', 'SPA', 'USA', 'ZIM'
            ], key=f"country_{i}")
            horse_type = st.selectbox("Type", [
                'Brown', 'Col', 'Filly', 'Gelding', 'Grey', 'Horse', 'Mare', 'Rig', 'Roan'
            ], key=f"type_{i}")
            surface = st.selectbox("Surface", ['TURF', 'ALL WEATHER'], key=f"surf_{i}")
            distance = st.number_input("Distance (meters)", min_value=800, max_value=2400, step=100, value=1200, key=f"dist_{i}")

        with col3:
            horse_place_perc = st.slider("Horse Placement %", 0.0, 1.0, 0.25, key=f"hpp_{i}")
            jockey_place_perc = st.slider("Jockey Placement %", 0.0, 1.0, 0.25, key=f"jpp_{i}")
            trainer_place_perc = st.slider("Trainer Placement %", 0.0, 1.0, 0.25, key=f"tpp_{i}")
            config = st.selectbox("Track Config", ['A', 'A+3', 'B', 'B+2', 'C', 'C+3'], key=f"config_{i}")
            going = st.selectbox("Track Condition", ['GOOD', 'YIELDING', 'FIRM', 'WET SLOW'], key=f"going_{i}")
            venue = st.selectbox("Venue", ['ST', 'HV'], key=f"venue_{i}")

        encoded_config = config_encoder.transform([[config]])[0][0]
        encoded_going = going_encoder.transform([[going]])[0][0]
        encoded_venue = venue_encoder.transform([venue])[0]
        encoded_country = horse_country_encoder.transform([horse_country])[0]
        encoded_type = horse_type_encoder.transform([horse_type])[0]
        encoded_surface = surface_encoder.transform([[surface]])[0][0]

        horse_dict = {
            'actual_weight': actual_wt,
            'config': encoded_config,
            'declared_weight': declared_wt,
            'distance': distance,
            'going': encoded_going,
            'horse_age': horse_age,
            'horse_country': encoded_country,
            'horse_place_perc': horse_place_perc,
            'horse_type': encoded_type,
            'jockey_place_perc': jockey_place_perc,
            'surface': encoded_surface,
            'trainer_place_perc': trainer_place_perc,
            'venue': encoded_venue
        }

        horses_data.append(horse_dict)
        horse_names.append(horse_name)

# ------------------------ Prediction and Visualization ------------------------

if st.button("Predict Finish Times"):
    input_df = pd.DataFrame(horses_data)
    input_scaled = scaler.transform(input_df)
    predictions = model.predict(input_scaled)

    result_df = pd.DataFrame({
        'Horse Name': horse_names,
        'Predicted Finish Time (s)': predictions
    }).sort_values(by='Predicted Finish Time (s)').reset_index(drop=True)

    result_df['Predicted Finish Time (s)'] = result_df['Predicted Finish Time (s)'].round(4)
    result_df['🏅'] = result_df.index.map(lambda x: '🥇' if x == 0 else ('🥈' if x == 1 else ('🥉' if x == 2 else '')))

    st.subheader("🏑 Predicted Rankings")
    st.dataframe(result_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=result_df, x='Predicted Finish Time (s)', y='Horse Name', s=150, color='mediumseagreen', ax=ax)

    for _, row in result_df.iterrows():
        ax.text(row['Predicted Finish Time (s)'] + 0.01, row['Horse Name'], f"{row['Predicted Finish Time (s)']:.4f}", va='center')

    ax.set_title("📊 Predicted Finish Times")
    ax.set_xlabel("Finish Time (seconds)")
    ax.set_ylabel("Horse")
    st.pyplot(fig)
