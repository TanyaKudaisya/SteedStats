import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved models and encoders
model = joblib.load("race_model.pkl")
scaler = joblib.load("scaler.pkl")
config_encoder = joblib.load("config_encoder.pkl")
going_encoder = joblib.load("going_encoder.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
horse_country_encoder = joblib.load("horse_country_encoder.pkl")
horse_type_encoder = joblib.load("horse_type_encoder.pkl")
surface_encoder = joblib.load("surface_encoder.pkl")

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
            horse_country = st.selectbox("Country", ['ARG', 'AUS', 'BRZ', 'CAN', 'FR', 'GB', 'GER', 'GR', 
 'IRE', 'ITY', 'JPN', 'NZ', 'SAF', 'SPA', 'USA', 'ZIM'], key=f"country_{i}")
            horse_type = st.selectbox("Type", ['Brown', 'Col', 'Filly', 'Gelding', 'Grey', 'Horse', 'Mare', 'Rig', 'Roan'], key=f"type_{i}")
            surface = st.selectbox("Surface", ['TURF', 'ALL WEATHER'], key=f"surf_{i}")
            distance = st.number_input("Distance (meters)",  min_value=800, max_value=2400, step=100, value=1200, key=f"dist_{i}")
        with col3:
            horse_place_perc = st.slider("Horse Placement %", 0.0, 1.0, 0.25, key=f"hpp_{i}")
            jockey_place_perc = st.slider("Jockey Placement %", 0.0, 1.0, 0.25, key=f"jpp_{i}")
            trainer_place_perc = st.slider("Trainer Placement %", 0.0, 1.0, 0.25, key=f"tpp_{i}")
            config = st.selectbox("Track Config", ['A', 'A+3', 'B', 'B+2', 'C', 'C+3' ], key=f"config_{i}")
            going = st.selectbox("Track Condition", ['GOOD', 'YIELDING', 'FIRM', 'WET SLOW'], key=f"going_{i}")
            venue = st.selectbox("Venue", ['ST', 'HV'], key=f"venue_{i}")

        # Encode categorical inputs
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

if st.button("Predict Finish Times"):
    input_df = pd.DataFrame(horses_data)
    input_scaled = scaler.transform(input_df)
    predictions = model.predict(input_scaled)

    # Build result DataFrame
    result_df = pd.DataFrame({
        'Horse Name': horse_names,
        'Predicted Finish Time (s)': predictions
    }).sort_values(by='Predicted Finish Time (s)').reset_index(drop=True)

    # Round finish times and add emoji ranking
    result_df['Predicted Finish Time (s)'] = result_df['Predicted Finish Time (s)'].round(4)
    result_df['🏅'] = result_df.index.map(lambda x: '🥇' if x == 0 else ('🥈' if x == 1 else ('🥉' if x == 2 else '')))

    # Display table
    st.subheader("🏁 Predicted Rankings")
    st.dataframe(result_df)

    # Enhanced chart using seaborn scatterplot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=result_df, x='Predicted Finish Time (s)', y='Horse Name', s=150, color='mediumseagreen', ax=ax)

    for i, row in result_df.iterrows():
        ax.text(row['Predicted Finish Time (s)'] + 0.01, row['Horse Name'], f"{row['Predicted Finish Time (s)']:.4f}", va='center')

    ax.set_title("📊 Predicted Finish Times")
    ax.set_xlabel("Finish Time (seconds)")
    ax.set_ylabel("Horse")
    st.pyplot(fig)
