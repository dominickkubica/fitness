import streamlit as st
import openai
import plotly.express as px
import pandas as pd
import numpy as np
import time
import os

###############################################################################
# 1. SETUP OPENAI (GPT-4)
###############################################################################

# Replace with your own key or reference it via an environment variable.
# For security, do NOT hardcode in production.
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

###############################################################################
# 2. GENERATIVE AI FUNCTIONS (GPT-4)
###############################################################################

def generate_meal_plan_gpt4(profile_data, mode="Casual"):
    """
    Generates a meal plan (including macros) using GPT-4 based on user profile data.
    mode can be "Casual" or "Advanced".
    """
    prompt = f"""
    You are an expert fitness coach. The user profile is as follows:
    Age: {profile_data.get('age')}
    Gender: {profile_data.get('gender')}
    Weight: {profile_data.get('weight')} lbs
    Height: {profile_data.get('height')} inches
    Activity Level: {profile_data.get('activity_level')}
    Fitness Goal: {profile_data.get('goal')}
    Dietary Preference: {profile_data.get('diet_preference')}
    
    Provide a {mode.lower()} meal plan for the next 7 days. 
    Include daily calorie targets, recommended macros (protein, carbs, fats), 
    and example foods per meal.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=700
    )
    
    return response["choices"][0]["message"]["content"]

def generate_workout_plan_gpt4(profile_data, mode="Casual"):
    """
    Generates a workout plan using GPT-4 based on user profile data.
    """
    prompt = f"""
    You are an expert fitness coach. The user profile is as follows:
    Age: {profile_data.get('age')}
    Gender: {profile_data.get('gender')}
    Weight: {profile_data.get('weight')} lbs
    Height: {profile_data.get('height')} inches
    Activity Level: {profile_data.get('activity_level')}
    Fitness Goal: {profile_data.get('goal')}
    
    Provide a {mode.lower()} workout plan for the next 7 days, including recommended 
    exercises, sets, reps, and any cardio/conditioning advice.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=700
    )
    
    return response["choices"][0]["message"]["content"]

def generate_water_intake_recommendation(profile_data):
    """
    A simple formula-based water intake recommendation.
    Using a rough heuristic: ~35-40 ml per kg bodyweight.
    """
    weight_kg = float(profile_data.get('weight', 150)) * 0.45359237
    recommended_ml = weight_kg * 40  # 40 ml per kg
    recommended_liters = recommended_ml / 1000.0
    
    return f"We recommend approximately {recommended_liters:.1f} liters of water per day."

def chat_with_gpt4(user_query: str):
    """
    Simple direct Q&A with GPT-4 (no external database).
    """
    prompt = f"""
    You are a highly knowledgeable fitness and nutrition AI assistant.
    Answer the user query accurately and helpfully:
    
    USER QUERY: {user_query}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

###############################################################################
# 3. STREAMLIT PAGES / WORKFLOW
###############################################################################

def page_user_profile():
    """Page for setting or updating user profile details."""
    st.title("Set / Update Your Profile")
    
    # Initialize session state for profile if not present
    if 'profile' not in st.session_state:
        st.session_state.profile = {}
    
    profile_data = st.session_state.profile
    
    age = st.number_input("Age", value=profile_data.get("age", 30), min_value=10, max_value=100)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                          index=["Male", "Female", "Other"].index(profile_data.get("gender", "Male")))
    weight = st.number_input("Weight (lbs)", value=profile_data.get("weight", 150), min_value=50, max_value=500)
    height = st.number_input("Height (inches)", value=profile_data.get("height", 65), min_value=48, max_value=90)
    activity_levels = ["Sedentary", "Light", "Moderate", "Intense"]
    activity_level = st.selectbox("Activity Level", activity_levels, 
                                  index=activity_levels.index(profile_data.get("activity_level", "Moderate")))
    goals = ["Cutting", "Bulking", "Maintenance"]
    goal = st.selectbox("Fitness Goal", goals, 
                        index=goals.index(profile_data.get("goal", "Maintenance")))
    diet_preferences = ["None", "Vegan", "Vegetarian", "Keto", "High Protein"]
    diet_preference = st.selectbox("Dietary Preference", diet_preferences,
                                   index=diet_preferences.index(profile_data.get("diet_preference", "None")))
    
    if st.button("Save Profile"):
        st.session_state.profile = {
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height,
            "activity_level": activity_level,
            "goal": goal,
            "diet_preference": diet_preference
        }
        st.success("Profile updated successfully!")

def page_dashboard():
    """
    Main user dashboard:  
    - Switch between Casual and Advanced modes  
    - View recommended meal plan, workout plan, water intake  
    - Visualize progress  
    - Chat with Fitness Coach (direct)
    """
    st.title("Fitness Coach Dashboard")
    
    # Ensure the user has set their profile
    if 'profile' not in st.session_state or not st.session_state.profile:
        st.info("Please set up your profile first.")
        page_user_profile()
        return
    
    profile_data = st.session_state.profile
    
    mode = st.radio("Choose Mode", ["Casual", "Advanced"], horizontal=True)
    st.write(f"**Current Mode:** {mode}")
    
    # Buttons to generate results
    if st.button("Generate Meal Plan"):
        with st.spinner("Generating meal plan with GPT-4..."):
            meal_plan = generate_meal_plan_gpt4(profile_data, mode=mode)
        st.subheader("Meal Plan")
        st.write(meal_plan)
    
    if st.button("Generate Workout Plan"):
        with st.spinner("Generating workout plan with GPT-4..."):
            workout_plan = generate_workout_plan_gpt4(profile_data, mode=mode)
        st.subheader("Workout Plan")
        st.write(workout_plan)
    
    st.subheader("Water Intake Recommendation")
    st.write(generate_water_intake_recommendation(profile_data))
    
    # Simple progress visualization (mock data)
    st.subheader("Progress Visualization")
    # Let's pretend we track 4 weeks of weight
    weeks = [f"Week {i}" for i in range(1, 5)]
    # Example: losing 2 lbs/week
    start_weight = profile_data["weight"]
    weight_data = np.linspace(start_weight, start_weight - 8, 4)
    df_progress = pd.DataFrame({"Week": weeks, "Weight": weight_data})
    fig = px.line(df_progress, x="Week", y="Weight", title="Weight Progress", markers=True)
    st.plotly_chart(fig)
    
    # Simple GPT-4 Q&A
    st.subheader("Ask the Fitness Coach (Direct GPT-4)")
    user_query = st.text_area("Type your question here", "")
    if st.button("Ask GPT-4"):
        if user_query.strip() == "":
            st.error("Please enter a question before asking.")
        else:
            with st.spinner("Generating answer..."):
                answer = chat_with_gpt4(user_query)
            st.write(answer)

###############################################################################
# 4. MAIN
###############################################################################

def main():
    st.set_page_config(page_title="Generative AI Fitness Coach", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio(
        "Go to",
        ["Profile", "Dashboard"]
    )
    
    if selection == "Profile":
        page_user_profile()
    elif selection == "Dashboard":
        page_dashboard()

if __name__ == "__main__":
    main()
