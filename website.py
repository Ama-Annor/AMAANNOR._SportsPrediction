import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and fitted scaler
try:
    model = joblib.load('best_enemble_model.pkl')
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

try:
    scaler = joblib.load('scaler.pkl')
    st.success("Scaler loaded successfully.")
except Exception as e:
    st.error(f"An error occurred while loading the scaler: {e}")

# App title and description
st.title('FIFA Player Rating Prediction')
st.write('Enter player attributes to predict their FIFA rating.')

# Function to get user inputs
def get_user_input():
    st.sidebar.header("Player Attributes")
    
    # List of all features used during training, excluding 'overall'
    features = ['height_cm', 'weight_kg', 'age', 'physic', 'power_strength',
                'power_jumping', 'movement_agility', 'movement_balance', 'dribbling',
                'skill_dribbling', 'skill_ball_control', 'shooting', 'passing',
                'skill_long_passing', 'skill_fk_accuracy', 'attacking_crossing',
                'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',
                'attacking_volleys', 'mentality_aggression', 'mentality_interceptions',
                'mentality_positioning', 'mentality_vision', 'mentality_penalties',
                'mentality_composure', 'movement_reactions', 'pace', 'movement_acceleration',
                'movement_sprint_speed', 'power_stamina', 'power_shot_power', 'power_long_shots',
                'defending', 'overall']

    user_data = {}
    for feature in features:
        user_data[feature] = st.sidebar.slider(f'{feature.replace("_", " ").title()}', 0, 100, 50)

    return pd.DataFrame(user_data, index=[0])

# Collect user input
user_input = get_user_input()

# Display user input
st.subheader('User Input Parameters')
st.write(user_input)

# Scale the user input and make prediction
try:
    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)
    
    # Display the prediction
    st.subheader('Prediction')
    st.write(f'The predicted FIFA player rating is: {prediction[0]:.2f}')
except Exception as e:
    st.error(f"An error occurred during scaling or prediction: {e}")

# Print feature names expected by the model
if hasattr(model, 'feature_names_in_'):
    st.write("Features expected by the model:", model.feature_names_in_)

st.balloons()
