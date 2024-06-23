import streamlit as st
import joblib
import pandas as pd

# Initialize model and scaler to None
model = None
scaler = None

# Load the model and fitted scaler
try:
    model = joblib.load('best_ensemble_model.pkl')
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

# Manually specify features based on your Jupyter notebook
features = [
    'age', 
    'height_cm', 
    'weight_kg', 
    'pace', 
    'shooting', 
    'passing', 
    'dribbling', 
    'defending', 
    'physic', 
    'attacking_crossing', 
    'attacking_finishing', 
    'attacking_heading_accuracy', 
    'attacking_short_passing', 
    'attacking_volleys', 
    'skill_dribbling', 
    'skill_fk_accuracy', 
    'skill_long_passing', 
    'skill_ball_control', 
    'movement_acceleration', 
    'movement_sprint_speed', 
    'movement_agility', 
    'movement_reactions', 
    'movement_balance', 
    'power_shot_power', 
    'power_jumping', 
    'power_stamina', 
    'power_strength', 
    'power_long_shots', 
    'mentality_aggression', 
    'mentality_interceptions', 
    'mentality_positioning', 
    'mentality_vision', 
    'mentality_penalties', 
    'mentality_composure'
]

# Function to get user inputs
def get_user_input(features):
    st.sidebar.header("Player Attributes")
    
    user_data = {}
    for feature in features:
        user_data[feature] = st.sidebar.slider(f'{feature.replace("_", " ").title()}', 0, 100, 50)

    return pd.DataFrame(user_data, index=[0])

# Collect user input
user_input = get_user_input(features)

# Display user input
st.subheader('User Input Parameters')
st.write(user_input)

# Scale the user input and make prediction
if model is not None and scaler is not None:
    try:
        scaled_input = scaler.transform(user_input)
        prediction = model.predict(scaled_input)
        
        # Display the prediction
        st.subheader('Prediction')
        st.write(f'The predicted FIFA player rating is: {prediction[0]:.2f}')
    except Exception as e:
        st.error(f"An error occurred during scaling or prediction: {e}")
        st.write("Error details:", str(e))
else:
    st.error("Model or scaler is not loaded properly. Please check the error messages above.")
    
st.balloons()
