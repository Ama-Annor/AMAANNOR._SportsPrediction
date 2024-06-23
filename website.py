import streamlit as st
import joblib
import pandas as pd

# Load the model and fitted scaler
try:
    model = joblib.load('best_ensemble_model.pkl')
    st.success("Model loaded successfully")
except ModuleNotFoundError as e:
    st.error(f"ModuleNotFoundError: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

try:
    scaler = joblib.load('scaler.pkl')
    st.success("Scaler loaded successfully")
except ModuleNotFoundError as e:
    st.error(f"ModuleNotFoundError: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

# App title and description
st.title('⚽ FIFA Prediction')
st.write('This is a simple FIFA prediction model. Please enter the required details to get the prediction.')

# Function to get user inputs
def training_attributes():
    st.sidebar.header("Player Attributes 📊")

    movement_reactions = st.sidebar.slider('Movement Reactions', 0, 100, 50)
    mentality_composure = st.sidebar.slider('Mentality Composure', 0, 100, 50)
    passing = st.sidebar.slider('Passing', 0, 100, 50)
    dribbling = st.sidebar.slider('Dribbling', 0, 100, 50)
    physic = st.sidebar.slider('Physic', 0, 100, 50)
    attacking_short_passing = st.sidebar.slider('Attacking Short Passing', 0, 100, 50)
    mentality_vision = st.sidebar.slider('Mentality Vision', 0, 100, 50)
    skill_long_passing = st.sidebar.slider('Skill Long Passing', 0, 100, 50)
    shooting = st.sidebar.slider('Shooting', 0, 100, 50)
    power_shot_power = st.sidebar.slider('Power Shot Power', 0, 100, 50)
    age = st.sidebar.slider('Age', 15, 50, 25)

    values = {
        'movement_reactions': movement_reactions, 
        'mentality_composure': mentality_composure,
        'passing': passing, 
        'dribbling': dribbling,             
        'physic': physic,                     
        'attacking_short_passing': attacking_short_passing,     
        'mentality_vision': mentality_vision,           
        'skill_long_passing': skill_long_passing,         
        'shooting': shooting,                   
        'power_shot_power': power_shot_power,           
        'age': age
    }

    characteristics = pd.DataFrame(values, index=[0])
    return characteristics

# Collect user input
user_input = training_attributes()

# Display user input
st.subheader('User Input Parameters 📋')
st.write(user_input)

# Scale the user input
try:
    scaled_input = scaler.transform(user_input)
    # Make prediction
    prediction = model.predict(scaled_input)

    # Display the prediction
    st.subheader('Prediction 🎯')
    st.write(f'The predicted value is: {prediction[0]}')
except Exception as e:
    st.error(f"An error occurred during scaling or prediction: {e}")

st.balloons()
