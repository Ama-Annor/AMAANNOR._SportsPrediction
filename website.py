import streamlit as st
import joblib
import pandas as pd

# Load the model and fitted scaler
try:
    model = joblib.load('best_enemble_model.pkl')
    st.success("Model loaded successfully.")
except ModuleNotFoundError as e:
    st.error(f"ModuleNotFoundError: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

try:
    scaler = joblib.load('scaler.pkl')
    st.success("Scaler loaded successfully.")
except ModuleNotFoundError as e:
    st.error(f"ModuleNotFoundError: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

# App title and description
st.title('⚽ FIFA Prediction ⚽')
st.write('This is a simple FIFA prediction model. Please enter the required details to get the prediction.')

# Function to get user inputs
def training_attributes():
    st.sidebar.header("Player Attributes")

    movement_reactions = st.sidebar.text_input('Movement Reactions', '50')
    mentality_composure = st.sidebar.text_input('Mentality Composure', '50')
    passing = st.sidebar.text_input('Passing', '50')
    dribbling = st.sidebar.text_input('Dribbling', '50')
    physic = st.sidebar.text_input('Physic', '50')
    attacking_short_passing = st.sidebar.text_input('Attacking Short Passing', '50')
    mentality_vision = st.sidebar.text_input('Mentality Vision', '50')
    skill_long_passing = st.sidebar.text_input('Skill Long Passing', '50')
    shooting = st.sidebar.text_input('Shooting', '50')
    power_shot_power = st.sidebar.text_input('Power Shot Power', '50')
    age = st.sidebar.text_input('Age', '25')

    values = {
        'movement_reactions': int(movement_reactions), 
        'mentality_composure': int(mentality_composure),
        'passing': int(passing), 
        'dribbling': int(dribbling),             
        'physic': int(physic),                     
        'attacking_short_passing': int(attacking_short_passing),     
        'mentality_vision': int(mentality_vision),           
        'skill_long_passing': int(skill_long_passing),         
        'shooting': int(shooting),                   
        'power_shot_power': int(power_shot_power),           
        'age': int(age)
    }

    characteristics = pd.DataFrame(values, index=[0])
    return characteristics

# Collect user input
user_input = training_attributes()

# Display user input
st.subheader('User Input Parameters')
st.write(user_input)

# Scale the user input
try:
    scaled_input = scaler.transform(user_input)
    # Make prediction
    prediction = model.predict(scaled_input)

    # Display the prediction
    st.subheader('Prediction')
    st.write(f'The predicted value is: {prediction[0]}')
except Exception as e:
    st.error(f"An error occurred during scaling or prediction: {e}")

# Optionally, use emojis for a more interactive experience
st.balloons()
