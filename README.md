# 🏇 SteedStats - Horse Race Finish Time Predictor

**SteedStats** is a machine learning-powered web app built with **Streamlit** that predicts the **finish times** of horses in a race. Given performance metrics, track details, and other parameters for multiple horses, the app ranks them based on expected race outcomes.

---

## 🚀 Features

- Predicts finish time (in seconds) for each horse using a trained ML model.
- Accepts input for multiple horses and displays:
  - Predicted time (rounded to 4 decimals)
  - Rank with medals 🥇🥈🥉
- Intuitive **scatter plot** comparison to visualize margins.
- Streamlit-powered, lightweight and interactive.

---

## 📊 Input Parameters

For each horse, the following fields are used:

- **Horse Name**
- **Horse Age**
- **Declared Weight** & **Actual Weight**
- **Horse Type** (e.g., Mare, Gelding, Horse, etc.)
- **Horse Country** (e.g., AUS, NZ, USA, etc.)
- **Track Configuration** (A, B, C, etc.)
- **Surface Type** (TURF / ALL WEATHER)
- **Track Condition (Going)** (GOOD, YIELDING, etc.)
- **Venue** (ST / HV)
- **Jockey, Trainer & Horse Placement %**
- **Race Distance (meters)**

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** for frontend
- **scikit-learn** for ML model
- **pandas**, **numpy**, **matplotlib**, **seaborn**
- **joblib** for model serialization

---

## 📦 Installation & Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/TanyaKudaisya/SteedStats.git
   cd SteedStats

2. **Create a virtual environment (optional but recommended):**:
  python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows

3. **Install requirements:**
  pip install -r requirements.txt

4. **Run the Streamlit app:**
  streamlit run app.py

**NOTE**
The model file race_model.pkl is not included due to GitHub's file size restrictions. You will need to add the trained model and encoder .pkl files ('going_encoder.pkl', 'horse_country_encoder.pkl', 'horse_type_encoder.pkl', 'race_model.pkl', 'scaler.pkl', 'surface_encoder.pkl', 'venue_encoder.pkl') manually into the project directory.

---

## 🧠 **Model & Dataset**
 - The model was trained on the HK Racing Dataset sourced from Kaggle (by gdaley), using features from both races.csv and runs.csv.
 - ML Algorithm used: RandomForestRegressor + optional VotingRegressor
 - Target: Actual Finish Time in seconds
