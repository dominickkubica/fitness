import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client correctly
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

###############################################################################
# 2. GENERATIVE AI FUNCTIONS (GPT-4)
###############################################################################


def generate_meal_plan_gpt4(profile_data, mode="Casual"):
    """
    Generates a meal plan using GPT-4 based on user profile data.
    """
    prompt = f"""
    You are a professional fitness coach. The user profile:
    
    - Age: {profile_data.get('age')}
    - Gender: {profile_data.get('gender')}
    - Weight: {profile_data.get('weight')} lbs
    - Height: {profile_data.get('height')} inches
    - Activity Level: {profile_data.get('activity_level')}
    - Fitness Goal: {profile_data.get('goal')}
    - Dietary Preference: {profile_data.get('diet_preference')}

    IMPORTANT: Create a meal plan that meets the calculated daily caloric needs of {profile_data.get('weight')} lbs.
    Structure the plan as 5 meals per day (Meal 1 through Meal 5) to ensure adequate caloric intake.
    
    Each meal should contribute to reaching the full daily caloric target of 3000-3500 calories.
    Do not underestimate portions or calories - this plan is for someone who needs higher caloric intake.

    Provide a concise but complete {mode.lower()} meal plan for 7 days. For each day:
    1. List each meal (Meal 1, Meal 2, Meal 3, Meal 4, Meal 5) with specific portions
    2. Include for EACH MEAL:
       - Total calories
       - Protein (g)
       - Carbohydrates (g)
       - Fats (g)
    3. Include DAILY TOTALS:
       - Total daily calories (ensure it meets 3000-3500 target)
       - Total daily protein (g)
       - Total daily carbohydrates (g)
       - Total daily fats (g)

    Distribute the calories across the 5 meals appropriately:
    - Meal 1: 20-25% of daily calories
    - Meal 2: 20-25% of daily calories
    - Meal 3: 20-25% of daily calories
    - Meal 4: 15-20% of daily calories
    - Meal 5: 15-20% of daily calories

    Be concise but ensure all 7 days are included with complete information.
    """

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )

    return response.choices[0].message.content


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
    
    IMPORTANT: Create a workout plan specifically designed for someone weighing {profile_data.get('weight')} lbs.
    Do not modify or assume different measurements than what is provided above.
    
    Provide a concise but complete {mode.lower()} workout plan for the next 7 days, 
    including recommended exercises, sets, reps, and any cardio/conditioning advice.
    """

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # Lowered from 0.7 to 0.2 for more consistent outputs
        max_tokens=2000,
    )

    return response.choices[0].message.content


def generate_water_intake_recommendation(profile_data):
    """
    A simple formula-based water intake recommendation.
    Using a rough heuristic: ~35-40 ml per kg bodyweight.
    """
    weight_kg = float(profile_data.get("weight", 150)) * 0.45359237
    recommended_ml = weight_kg * 40  # 40 ml per kg
    recommended_liters = recommended_ml / 1000.0

    return (
        f"We recommend approximately {recommended_liters:.1f} liters of water per day."
    )


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
        model="gpt-4-0125-preview",  # Using GPT-4 Turbo for faster responses
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,  # Increased for more complete answers
    )
    return response.choices[0].message.content


###############################################################################
# 3. STREAMLIT PAGES / WORKFLOW
###############################################################################


def page_user_profile():
    """Page for setting or updating user profile details."""
    st.title("Set / Update Your Profile")

    # Initialize session state for profile if not present
    if "profile" not in st.session_state:
        st.session_state.profile = {}

    profile_data = st.session_state.profile

    age = st.number_input(
        "Age", value=profile_data.get("age", 30), min_value=10, max_value=100
    )
    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"],
        index=["Male", "Female", "Other"].index(profile_data.get("gender", "Male")),
    )
    weight = st.number_input(
        "Weight (lbs)",
        value=profile_data.get("weight", 150),
        min_value=50,
        max_value=500,
    )
    height = st.number_input(
        "Height (inches)",
        value=profile_data.get("height", 65),
        min_value=48,
        max_value=90,
    )
    activity_levels = ["Sedentary", "Light", "Moderate", "Intense"]
    activity_level = st.selectbox(
        "Activity Level",
        activity_levels,
        index=activity_levels.index(profile_data.get("activity_level", "Moderate")),
    )
    goals = ["Cutting", "Bulking", "Maintenance"]
    goal = st.selectbox(
        "Fitness Goal",
        goals,
        index=goals.index(profile_data.get("goal", "Maintenance")),
    )
    diet_preferences = ["None", "Vegan", "Vegetarian", "Keto", "High Protein"]
    diet_preference = st.selectbox(
        "Dietary Preference",
        diet_preferences,
        index=diet_preferences.index(profile_data.get("diet_preference", "None")),
    )

    if st.button("Save Profile"):
        st.session_state.profile = {
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height,
            "activity_level": activity_level,
            "goal": goal,
            "diet_preference": diet_preference,
        }
        st.success("Profile updated successfully!")


def calculate_weight_projection(profile_data, meal_plan_calories=None):
    """
    Calculate weight projection based on goal and calories.
    Returns weekly weights for 12 weeks.
    """
    start_weight = profile_data["weight"]
    goal = profile_data["goal"]

    # Calculate weekly change based on goal
    if meal_plan_calories:
        # Calculate TDEE (Total Daily Energy Expenditure) - simplified estimate
        activity_multipliers = {
            "Sedentary": 1.2,
            "Light": 1.375,
            "Moderate": 1.55,
            "Intense": 1.725,
        }
        # Basic BMR calculation using Harris-Benedict equation
        if profile_data["gender"] == "Male":
            bmr = (
                66
                + (6.23 * start_weight)
                + (12.7 * profile_data["height"])
                - (6.8 * profile_data["age"])
            )
        else:
            bmr = (
                655
                + (4.35 * start_weight)
                + (4.7 * profile_data["height"])
                - (4.7 * profile_data["age"])
            )

        tdee = bmr * activity_multipliers[profile_data["activity_level"]]
        # Calculate daily caloric deficit/surplus
        daily_calorie_difference = meal_plan_calories - tdee
        # 1 pound of fat ≈ 3500 calories
        weekly_weight_change = daily_calorie_difference * 7 / 3500
    else:
        # Default projections if no meal plan calories available
        weekly_weight_change = {
            "Cutting": -1,  # 1 lb loss per week
            "Bulking": 0.5,  # 0.5 lb gain per week
            "Maintenance": 0,  # No change
        }.get(goal, 0)

    # Generate weekly weights for 12 weeks
    weeks = [f"Week {i}" for i in range(13)]  # 0 to 12 weeks
    weight_data = [start_weight + (weekly_weight_change * i) for i in range(13)]

    return weeks, weight_data


def page_dashboard():
    """
    Main user dashboard with enhanced visualization
    """
    st.title("Fitness Coach Dashboard")

    # Ensure the user has set their profile
    if "profile" not in st.session_state or not st.session_state.profile:
        st.info("Please set up your profile first.")
        page_user_profile()
        return

    profile_data = st.session_state.profile

    mode = st.radio("Choose Mode", ["Casual", "Advanced"], horizontal=True)
    st.write(f"**Current Mode:** {mode}")

    # Store meal plan and calories in session state
    if "meal_plan" not in st.session_state:
        st.session_state.meal_plan = None
        st.session_state.daily_calories = None

    # Buttons to generate results
    if st.button("Generate Meal Plan"):
        with st.spinner("Generating meal plan with GPT-4..."):
            meal_plan = generate_meal_plan_gpt4(profile_data, mode=mode)
            st.session_state.meal_plan = meal_plan

            # Extract daily calories from the meal plan using a simple average
            try:
                # Look for daily total calories in the response
                import re

                calorie_matches = re.findall(
                    r"Total daily calories:?\s*(\d+)", meal_plan
                )
                if calorie_matches:
                    st.session_state.daily_calories = sum(
                        map(int, calorie_matches)
                    ) / len(calorie_matches)
            except Exception:
                st.session_state.daily_calories = None

        st.subheader("Meal Plan")
        st.write(meal_plan)

    if st.button("Generate Workout Plan"):
        with st.spinner("Generating workout plan with GPT-4..."):
            workout_plan = generate_workout_plan_gpt4(profile_data, mode=mode)
        st.subheader("Workout Plan")
        st.write(workout_plan)

    st.subheader("Water Intake Recommendation")
    st.write(generate_water_intake_recommendation(profile_data))

    # Enhanced progress visualization
    st.subheader("Weight Projection (12 Weeks)")
    weeks, weight_data = calculate_weight_projection(
        profile_data,
        (
            st.session_state.daily_calories
            if hasattr(st.session_state, "daily_calories")
            else None
        ),
    )

    df_progress = pd.DataFrame({"Week": weeks, "Weight": weight_data})

    fig = px.line(
        df_progress,
        x="Week",
        y="Weight",
        title=f"Weight Projection Based on {profile_data['goal']} Goal",
        markers=True,
    )

    # Add annotations
    fig.add_annotation(
        text=f"Goal: {profile_data['goal']}",
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
    )

    if st.session_state.daily_calories:
        fig.add_annotation(
            text=f"Target Daily Calories: {int(st.session_state.daily_calories)}",
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.92,
            showarrow=False,
        )

    # Customize the layout
    fig.update_layout(
        xaxis_title="Timeline", yaxis_title="Weight (lbs)", hovermode="x unified"
    )

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
    selection = st.sidebar.radio("Go to", ["Profile", "Dashboard"])

    if selection == "Profile":
        page_user_profile()
    elif selection == "Dashboard":
        page_dashboard()


if __name__ == "__main__":
    main()
