import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import openai
from database_handler import load_database, update_database  # Import the database functions
import io


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


copy_df = df.copy()    
# Add a new column 'Goal Met' based on the calorie goal
copy_df['Goal Met'] = copy_df['Net Calories'].apply(lambda x: 'Yes' if x <= 1300 else 'No')
copy_df = copy_df.sort_values('Date', ascending = True)

# defining function for icon UI
def generate_svg_icon(day, goal_met):
    """Generate an SVG circle icon with a ring and 'Day N' label inside."""
    color = "green" if goal_met == "Yes" else "red"
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
# Function to display progress icons and analysis
def perform_analysis():
    display_progress_icons(copy_df)  # Assuming this function exists

    # Center-aligned subheading
    st.divider()
    st.write('### What do these numbers mean?')

    total_calories_burned = copy_df['Calories Burned'].sum()
    total_distance_travelled = copy_df['Distance Travelled (m)'].sum()

    # Load fast food dataset
    food_data = pd.read_csv('data/fastfood.csv')
    random_food = food_data.sample(1).iloc[0]
    food_name = random_food['item']
    food_brand = random_food['restaurant']
    food_calories = random_food['calories']

    equivalent_quantity = total_calories_burned / food_calories

    def compare_distance_to_landmarks(total_distance):
    # Define a detailed prompt to guide the API for varied, accurate comparisons
        prompt = (
            f"The total distance traveled is {total_distance} meters. "
            "Provide a fun, relatable comparison using well-known, varied distances such as:\n"
            "- Golden Gate Bridge (2,737 meters) as 'walking across the Golden Gate Bridge X times'\n"
            "- Length of a football field (100 meters) as 'walking X football fields'\n"
            "- Brooklyn Bridge (1,834 meters) as 'walking the Brooklyn Bridge X times'\n"
            "- Central Park loop (9,656 meters) as 'completing X laps around Central Park'\n"
            "- Times Square to Empire State Building (650 meters) as 'walking from Times Square to the Empire State Building X times'\n"
            "Output just the comparison, and make sure to vary it for user engagement. Avoid repetitive landmark usage. Also your output should"
            " only and strictly be in the grammatical form of 'walking across X bridge X times', or 'completing x laps around x landmark' Dont"
            "take this literally, but maintain the grammatical form of sentence I just showed you. Also end with only an exclamation mark, no full stop before it"
        )
        
        try:
            # Make the API call to OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
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
        
    def plot_non_cumulative_distance(df):
        # Ensure 'Date' is in datetime format for accurate plotting
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date to maintain correct order
        df = df.sort_values('Date')
        
        # Plot non-cumulative distance over time
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['Distance Travelled (m)'], marker='o', color='blue', linewidth=2, markersize=5)
        plt.title('Daily Distance Travelled Over Time')
        plt.xlabel('Date')
        plt.ylabel('Distance Travelled (m)')
        plt.grid(True)
        plt.gca().tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-axis values
        plt.tight_layout()
        plt.show()
        
        # Display plot in Streamlit
        st.pyplot(plt.gcf())
        

    comparison_result = compare_distance_to_landmarks(total_distance_travelled)

    # Display contextualized insights
    st.write(f"###### You burned the equivalent of eating {equivalent_quantity:.1f} {food_name}s from {food_brand} in the past week!")
    st.write("##### OR")
    st.write("###### If we needed to generate the amount of calories you burned in the last week with a solar panel, it would take...")
    st.write("###### 1 ☀️ = energy generated by a solar panel in 1 hour")
    num_panels = round(total_calories_burned / 71.4)
    st.write("☀️" * num_panels)
    st.write(f"##### {num_panels} hours!")
    st.divider()


    st.write(f"#### You also ran the equivalent of {comparison_result}")

    st.write("##### Non-cumulative Distance Over Time")
    plot_non_cumulative_distance(copy_df)

    # visualization for distance contextualization


# Buttons
# Button for Analyze and its content
analyze_clicked = st.button("Analyze", key="analyze_button")

if analyze_clicked:
    with st.expander("Analysis Results", expanded=True):
        perform_analysis()


# Function to analyze data with GPT
def analyze_with_gpt(df, goal):
    # Convert DataFrame to CSV string
    df_string = df.to_csv(index=False)
    
    prompt = f"""
    Here is a dataset of the user's last 7 days' health data in CSV format:
    
    {df_string}
    
    The user's net calorie goal is {goal} kcal.
    I want 3 points from you. one sentence on each and succinct please. short and sweet sentences.
    Please analyze the dataset and:
    1. Provide insights on their calorie management and exercise habits.
    2. Suggest actionable improvements.
    3. give the user some motivation and positive reinforcement on how well they are doing (if at all), and give them hope for next week. 
    keep all these sentences limited to 2 ideas/sentence at most.
    talk in second person, like you are talking to the user themselves
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful health and fitness data analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        return response['choices'][0]['message']['content']
    
    except Exception as e:
        st.error(f"OpenAI API call failed: {e}")
        return None

recommended_net_calorie = 1300 # hardcoding ;) (for now) 

# Function to simulate chatbot response
def chatbot_response(user_input):
    # Replace this with your model or API call
    response = f"Echo: {user_input}"
    return response

last_7_days = copy_df.tail(7)
print(last_7_days)
# Add chatbot within the Discover Insights section
if st.button("Discover insights", key="discover_insights_button"):
    with st.expander("Discover Insights", expanded=True):
        st.write('### This is how you could have improved')
        
        # Get last 7 days of data
        analysis_result = analyze_with_gpt(last_7_days, recommended_net_calorie)
        
        if analysis_result:
            st.write("### Analysis of workout patterns:")
            st.write(analysis_result)
            
            # Chatbot functionality
            st.write("### Talk to your personal health assistant:")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            user_input = st.text_input("You:", key="user_input")
            if st.button("Send"):
                if user_input:
                    # Get chatbot response
                    response = chatbot_response(user_input)
                    # Update chat history
                    st.session_state.chat_history.append({"user": user_input, "bot": response})
                    st.session_state.user_input = ""  # Clear input field after sending

            for chat in st.session_state.chat_history:
                st.write(f"You: {chat['user']}")
                st.write(f"Bot: {chat['bot']}")

num_achieved = np.sum(last_7_days["Goal Met"] == "Yes")

# describing the cheat meal recommendation function
def get_cheat_meal():
    prompt = (
        "Recommend a cheat meal that does not exceed 800 calories. "
        "Make it flavorful and fun, and go nuts with the options. "
        "Provide multiple options. THe output should be bullet points with only the cheat meal recs and cals, example:"
        "- big whopper, fries, and a diet coke (750 cals)"
        "- rib-eye steak with mashed potatoes and romulan ale"
        "essentally you get the gist right. just the bullet points nothing else."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    
    # Extract response text
    return response['choices'][0]['message']['content']

if num_achieved > 3:
    st.write("### Congratulations! You have been rewarded with a cheat meal because of your consistency!")
    st.write("Here\'s what we recommend you have (feel free to break free ;) :  ")
    cheat_meal_rec = get_cheat_meal()
    st.write(cheat_meal_rec)
