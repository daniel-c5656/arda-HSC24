import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import openai
from database_handler import load_database, update_database  # Import the database functions

# Load the calorie and workout database
csv_path = 'data/main_data.csv'
df = load_database(filepath=csv_path)  # Load or initialize the database

st.write("""
# Welcome to ARDA!
### Hope your day is going great so far. Stay healthy!
""")

st.write("## What did you eat today? Did you remember to move?")

col1, col2, col3 = st.columns(3)

with col1:
    workout_type = st.selectbox("Workout type", ("walking", "running"))
    distance_moved = st.number_input("Distance travelled (m)", min_value=0, step=100)
with col2:
    calories_burned = st.number_input("Calories burned", min_value=0, step=50)
    workout_duration = st.number_input("Workout length (in minutes)", min_value=0, step=5)
with col3:
    steps = st.number_input("Steps taken", min_value=0, step=500)
    date = st.date_input("Date", value=datetime.date.today())
    time_intervals = [(datetime.datetime.min + datetime.timedelta(minutes=15 * i)).time()
    for i in range(96)]  # 96 intervals in a day (24 * 4 = 96)

    # Use selectbox to let the user choose a time from these intervals
    time = st.selectbox("Select Time", time_intervals, index=32)


# Meal inputs
st.header("Enter your meals:")
breakfast = st.text_input("Breakfast (e.g., 2 eggs, 3 strips of bacon):")
lunch = st.text_input("Lunch (e.g., chicken sandwich, salad):")
dinner = st.text_input("Dinner (e.g., steak, mashed potatoes):")


# openai api key (DESTROY BEFORE MAKING REPO PUBLIC)
openai.api_key = "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"

# calculating total calories consumed on a particular day
def calculate_total_calories(breakfast, lunch, dinner, custom_prompt=None):
    # Use a custom prompt if provided, otherwise use default
    prompt = (
        custom_prompt
        if custom_prompt
        else f"Calculate the total calorie intake for the day based on these meals:\n"
             f"Breakfast: {breakfast}\n"
             f"Lunch: {lunch}\n"
             f"Dinner: {dinner}\n"
             f"Provide the total calories as a single integer. Just give me an integer value. Nothing else. Always overestimate rather than underestimate."
    )
    
    # API call to OpenAI's GPT model
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if available
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        # Extract the response content
        model_response = response['choices'][0]['message']['content'].strip()
        st.write("Model Response:", model_response)  # Log the response for debugging
        
        # Extract the integer from the model's response
        total_calories = ''.join(filter(str.isdigit, model_response))
        return int(total_calories) if total_calories else "Error: No valid calorie value found."
    
    except Exception as e:
        st.error(f"OpenAI API call failed: {e}")
        return "Error: API call failed."


# declaring the prompt for the API call
custom_prompt = (
    f"Calculate the total calorie intake for the day based on the following meals:\n"
    f"Breakfast: {breakfast}\n"
    f"Lunch: {lunch}\n"
    f"Dinner: {dinner}\n\n"
    f"Provide the total calories as a single integer value. No explanation. Always overestimate, rather than underestimate."
)

# declaring a variable to store the result
total_calories = None

# Checks if calories have been successfully calculated before allowing the data to be recorded.
calculated = False
# calculating the total cals by using the calculate_total_calories function
if st.button("Calculate"):
    try:
        total_calories = calculate_total_calories(breakfast, lunch, dinner, custom_prompt)
        st.success(f"Your total calorie intake for the day is: {total_calories} kcal")
        calculated = True
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display total calories below the button
if total_calories is not None:
    st.write(f"**Total Calories:** {total_calories} kcal")

# Record Data
if st.button("Record data", disabled=(not calculated)):
    total_calories = calculate_total_calories(breakfast, lunch, dinner)  # Calculate total calories
    combined_date_time = datetime.datetime.combine(date, time)  # Keep both date and time
    
    # Update the database with all user inputs
    update_database(
        date_str=combined_date_time.strftime('%Y-%m-%d %H:%M:%S'),  # Use combined date-time string
        workout_type=workout_type,
        calories_burned=calories_burned,
        distance=distance_moved,
        steps=steps,
        workout_length=workout_duration,
        breakfast=breakfast,
        lunch=lunch,
        dinner=dinner,
        calories_consumed=total_calories,
        filepath=csv_path  # Use your CSV path
    )

    st.success("Data recorded successfully!")
    calculated = False
    # Reload the database after saving new data
    df = load_database(filepath=csv_path)


# Display updated data below only once
st.write("## Your Workout and Meal Data:")
st.dataframe(df.sort_values("Date", ascending=True))


#============================================================
# PERFORMING WEEKLY ANALYSIS NOW
#============================================================

st.title("Weekly Data Analysis")

# creating a copy of df with additional column of "yes" or "no" for daily goal
def evaluate_daily_goal(df, calorie_goal=1300):
    """
    Create a copy of the DataFrame and add a 'Goal Met' column based on the net calorie goal.

    Parameters:
    df (pd.DataFrame): The original DataFrame with a 'Net Calories' column.
    calorie_goal (int): The daily net calorie goal (default is 1300).

    Returns:
    pd.DataFrame: A copy of the original DataFrame with an additional 'Goal Met' column.
    """
    # Create a copy of the DataFrame
    copy_df = df.copy()
    
    # Add a new column 'Goal Met' based on the calorie goal
    copy_df['Goal Met'] = copy_df['Net Calories'].apply(lambda x: 'Yes' if x <= calorie_goal else 'No')
    copy_df = copy_df.sort_values('Date', ascending = True)
    return copy_df


# defining function for icon UI
def generate_svg_icon(day, goal_met):
    """Generate an SVG circle icon with a ring and 'Day N' label inside."""
    color = "green" if goal_met == "Yes" else "yellow"
    fill_ratio = 1 if goal_met == "Yes" else 0.5  # Full or half-filled ring

    svg = f"""
    <svg width="140" height="140">
        <!-- Outer ring -->
        <circle cx="70" cy="70" r="60" stroke="{color}" stroke-width="6" fill="none" />
        
        <!-- Inner filled circle with hollow space -->
        <circle cx="70" cy="70" r="{int(60 * 0.8)}" fill="{color}" fill-opacity="0.5" />
        
        <!-- Day number in the center -->
        <text x="50%" y="55%" text-anchor="middle" font-size="18" fill="black">Day {day}</text>
    </svg>
    """
    return svg

def display_progress_icons(copy_df):
    st.write("### Weekly Progress")
    
    last_7_days = copy_df.tail(7)
    rows = [last_7_days.iloc[i:i+4] for i in range(0, len(last_7_days), 4)]

    for row in rows:
        columns = st.columns(4, gap="large")  # Dynamically adjust to maintain center alignment
        offset = (4 - len(row)) // 2  # Calculate offset for centering the last row
        
        for i, (index, record) in enumerate(row.iterrows()):
            day = record['Date'].strftime('%d')
            goal_met = record['Goal Met']
            net_calories = record['Net Calories']
            svg_icon = generate_svg_icon(day, goal_met)

            # Use offset on the last row for centering
            with columns[i + offset]:
                st.components.v1.html(svg_icon, height=160, width=160)
                st.markdown(f"**Net Calories**: {net_calories} kcal", unsafe_allow_html=True)


copy_df = evaluate_daily_goal(df)


# ANALYZING WEEKLY DATA (USER PRESSES BUTTON NOW)
if st.button("Analyze"):
    display_progress_icons(copy_df)

    # Center-aligned subheading
    st.markdown(
        """
        <div style='text-align: center; font-size: 28px; font-weight: bold; margin-top: 20px;'>
            Here's what your stats over the past week signify:
        </div>
        """,
        unsafe_allow_html=True
    )

    # CONTEXTUALIZING THE DATA

    total_calories_burned = copy_df['Calories Burned'].sum()
    total_distance_travelled = copy_df['Distance Travelled (m)'].sum()
    food_data = pd.read_csv('data/fastfood.csv')
    # Randomly select a food item from the dataset
    random_food = food_data.sample(1).iloc[0]
    food_name = random_food['item']
    food_brand = random_food['restaurant']
    food_calories = random_food['calories']

    # Calculate the equivalent number of food items burned
    equivalent_quantity = total_calories_burned / food_calories
    def compare_distance_to_landmarks(total_distance):
    # Define the prompt for OpenAI API
        prompt = (
            f"The total distance traveled is {total_distance} meters. "
            "Compare this distance to famous landmarks or multiples of distances between famous cities. "
            "Provide a comparison like 'This is equivalent to walking across the Golden Gate Bridge 5 times' or "
            "'This is equivalent to the distance between New York and Washington D.C.' THe output should be just 'walking across the Golden Gate Bridge 5 times'"
            " or 'the distance between New York and Washington D.C.' Thats the output you should give, and nothing else."
        )
        
        try:
            # Make the API call to OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use your existing model
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}],
                max_tokens=100
            )
            
            # Extract the response
            comparison = response['choices'][0]['message']['content'].strip()
            return comparison
        
        except Exception as e:
            st.error(f"OpenAI API call failed: {e}")
            return "Error: Could not retrieve comparison."
    comparison_result = compare_distance_to_landmarks(total_distance_travelled)

    # Return the contextualized statement
    st.write(f"You burned the equivalent of eating {equivalent_quantity:.1f} {food_name}s from {food_brand}!!")
    st.write(f"You also ran the equivalent of {comparison_result}!")
