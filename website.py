import streamlit as st
import joblib
import pandas as pd

#Load the model and fitted scaler
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

st.title('⚽ FIFA Prediction')
st.write('This is a simple FIFA prediction model. Please enter the required details to get the prediction.')

#Inputs from user
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

user_input = training_attributes()

#Input is displayed
st.subheader('User Input Parameters 📋')
st.write(user_input)

#Input is scaled
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


# Sources:
# https://charumakhijani.medium.com/machine-learning-model-deployment-as-a-web-app-using-streamlit-4e542d0adf15
#https://www.bing.com/ck/a?!&&p=7bc552995dccd74bJmltdHM9MTcxOTEwMDgwMCZpZ3VpZD0xMzY1MmFhZC0xYTVmLTY2ZmMtMTY4ZS0zODM0MWIzODY3ZDImaW5zaWQ9NTIxMQ&ptn=3&ver=2&hsh=3&fclid=13652aad-1a5f-66fc-168e-38341b3867d2&psq=deploy+streamlit+app+using+github&u=a1aHR0cHM6Ly9kb29kbGVjbG91ZHMubWVkaXVtLmNvbS91c2luZy1naXRodWItcGFnZXMtdG8taG9zdC15b3VyLXN0cmVhbWxpdC1hcHAtZjI3NGNiZTNiM2Fm&ntb=1
#How to make my app lively - https://www.bing.com/ck/a?!&&p=049528afa5e6db9aJmltdHM9MTcxOTEwMDgwMCZpZ3VpZD0xMzY1MmFhZC0xYTVmLTY2ZmMtMTY4ZS0zODM0MWIzODY3ZDImaW5zaWQ9NTIwNg&ptn=3&ver=2&hsh=3&fclid=13652aad-1a5f-66fc-168e-38341b3867d2&psq=how+to+add+emojis+and+pictures+to+your+streamlit+from+github&u=a1aHR0cHM6Ly9naXRodWIuY29tL3N0cmVhbWxpdC9lbW9qaS1zaG9ydGNvZGVz&ntb=1
