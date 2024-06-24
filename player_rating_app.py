import streamlit as st
import pandas as pd
import joblib

# Loading model into python script
model = joblib.load('model_best.pkl')

# features used in trained model
expected_features = ['movement_reactions', 'mentality_composure', 'passing', 'dribbling', 'physic',
                     'attacking_short_passing', 'mentality_vision', 'skill_long_passing', 'shooting',
                     'power_shot_power', 'age']

# shows stars on website
def compute_stars(rating):
    if rating <= 0:
        return 0.00
    elif rating >= 5:
        return 5.00
    else:
        return rating

def main():
    st.title("⚽FIFA Player Rating Predictor🏆")
    
    html_temp = """
    <div style="background-color:#025246;padding:10px;border-radius:10px;">
    <h2 style="color:white;text-align:center;">Player Rating Predictor App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.markdown("### Enter player attributes to predict the overall rating 🌟")

    st.markdown(
        """
        <style>
        .stSlider label {
            font-size: 1.2rem;
        }
        .stNumberInput label {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Slider and text entries
    movement_reactions_slider = st.slider("Movement Reactions", min_value=0, max_value=100, step=1, key="movement_reactions_slider")
    movement_reactions = float(movement_reactions_slider)

    mentality_composure_slider = st.slider("Mentality Composure", min_value=0, max_value=100, step=1, key="mentality_composure_slider")
    mentality_composure = float(mentality_composure_slider)

    passing_slider = st.slider("Passing", min_value=0, max_value=100, step=1, key="passing_slider")
    passing = float(passing_slider)

    dribbling_slider = st.slider("Dribbling", min_value=0, max_value=100, step=1, key="dribbling_slider")
    dribbling = float(dribbling_slider)

    physic_slider = st.slider("Physical", min_value=0, max_value=100, step=1, key="physic_slider")
    physic = float(physic_slider)

    attacking_short_passing_slider = st.slider("Short Passing", min_value=0, max_value=100, step=1, key="attacking_short_passing_slider")
    attacking_short_passing = float(attacking_short_passing_slider)

    mentality_vision_slider = st.slider("Mentality Vision", min_value=0, max_value=100, step=1, key="mentality_vision_slider")
    mentality_vision = float(mentality_vision_slider)

    skill_long_passing_slider = st.slider("Long Passing", min_value=0, max_value=100, step=1, key="skill_long_passing_slider")
    skill_long_passing = float(skill_long_passing_slider)

    shooting_slider = st.slider("Shooting", min_value=0, max_value=100, step=1, key="shooting_slider")
    shooting = float(shooting_slider)

    power_shot_power_slider = st.slider("Shot Power", min_value=0, max_value=100, step=1, key="power_shot_power_slider")
    power_shot_power = float(power_shot_power_slider)

    age_slider = st.slider("Age", min_value=0, max_value=50, step=1, key="age_slider")
    age = float(age_slider)

    actual_rating = st.number_input("Enter the Actual Rating", min_value=0.0, max_value=100.0, step=0.01, value=4.00)

    if st.button("Predict"):
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
        predicted_rating = prediction[0]
        stars = compute_stars(predicted_rating)

        # Visual representation of stars
        full_stars = int(stars)
        remainder = stars - full_stars

        star_icons = '⭐' * full_stars
        if remainder > 0:
            if remainder < 0.25:
                star_icons += '☆'
            elif remainder < 0.75:
                star_icons += '½'
            else:
                star_icons += '★'
                
        # Confidence level calculation using user-provided actual rating
        confidence = 1 - abs(predicted_rating - actual_rating) / actual_rating
        confidence_level = f"{confidence * 100:.2f}%"

        st.success(f'Predicted Player Rating (Overall): {star_icons} ({stars:.2f})')
        st.success(f'Confidence: {confidence_level}')

if __name__ == '__main__':
    main()

