import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model_best.pkl')

expected_features = ['movement_reactions', 'mentality_composure', 'passing', 'dribbling', 'physic',
                     'attacking_short_passing', 'mentality_vision', 'skill_long_passing', 'shooting',
                     'power_shot_power', 'age']

def main():
    st.title("⚽FIFA Player Rating Predictor🏆")
    
    html_temp = """
    <div style="background-color:#025246;padding:10px;border-radius:10px;">
    <h2 style="color:white;text-align:center;">Player Rating Predictor App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    
    st.markdown("### Enter player attributes to predict the overall rating 🌟")

    movement_reactions = st.number_input("Movement Reactions ⚡", min_value=0.0)
    mentality_composure = st.number_input("Mentality Composure 🧠", min_value=0.0)
    passing = st.number_input("Passing 🎯", min_value=0.0)
    dribbling = st.number_input("Dribbling 🕺", min_value=0.0)
    physic = st.number_input("Physic 💪", min_value=0.0)
    attacking_short_passing = st.number_input("Attacking Short Passing 🔄", min_value=0.0)
    mentality_vision = st.number_input("Mentality Vision 👁️", min_value=0.0)
    skill_long_passing = st.number_input("Skill Long Passing 🦵", min_value=0.0)
    shooting = st.number_input("Shooting 🎯", min_value=0.0)
    power_shot_power = st.number_input("Power Shot Power 💥", min_value=0.0)
    age = st.number_input("Age 📅", min_value=0.0)

    if st.button("Predict 🧙‍♂️"):
        features = {
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
        
        df = pd.DataFrame([features], columns=expected_features)

        prediction = model.predict(df)
        output = prediction[0]

        st.success(f'Predicted Player Rating (Overall): {output:.2f} ⭐')

if __name__ == '__main__':
    main()

#Sources:
#https://www.bing.com/ck/a?!&&p=f0f409d84b6f759dJmltdHM9MTcxOTEwMDgwMCZpZ3VpZD0xMzY1MmFhZC0xYTVmLTY2ZmMtMTY4ZS0zODM0MWIzODY3ZDImaW5zaWQ9NTIwNw&ptn=3&ver=2&hsh=3&fclid=13652aad-1a5f-66fc-168e-38341b3867d2&psq=how+to+add+emojis+and+pictures+to+your+streamlit+&u=a1aHR0cHM6Ly93d3cucmVzdGFjay5pby9kb2NzL3N0cmVhbWxpdC1rbm93bGVkZ2Utc3RyZWFtbGl0LWVtb2ppLWd1aWRl&ntb=1
#https://www.bing.com/ck/a?!&&p=afff13c240e160edJmltdHM9MTcxOTEwMDgwMCZpZ3VpZD0xMzY1MmFhZC0xYTVmLTY2ZmMtMTY4ZS0zODM0MWIzODY3ZDImaW5zaWQ9NTQ2OA&ptn=3&ver=2&hsh=3&fclid=13652aad-1a5f-66fc-168e-38341b3867d2&psq=using+a+python+file+and+power+shell+to+streamlit&u=a1aHR0cHM6Ly93d3cucmVzdGFjay5pby9kb2NzL3N0cmVhbWxpdC1rbm93bGVkZ2UtcnVuLXN0cmVhbWxpdC1hcHAtanVweXRlci1ub3RlYm9vayM6fjp0ZXh0PVRvJTIwcnVuJTIwYSUyMFN0cmVhbWxpdCUyMGFwcCUyMGluJTIwYSUyMEp1cHl0ZXIsY29tbWFuZCUyMGZyb20lMjB3aXRoaW4lMjB0aGUlMjBub3RlYm9vayUzQSUyMCUyMXN0cmVhbWxpdCUyMHJ1biUyMHlvdXJfc3RyZWFtbGl0X2FwcC5weQ&ntb=1
