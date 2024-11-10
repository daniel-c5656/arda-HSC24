import pandas as pd
import os
import streamlit as st

# Function to initialize or load the database
def load_database(filepath='/Users/arnavagrawal/Documents/arda-HSC24/data/main_data.csv'):
    # Check if the file exists
    if os.path.exists(filepath):
        if os.path.getsize(filepath) == 0:
            # Initialize empty database with headers
            st.warning("Empty file detected. Initializing headers.")
            columns = [
                'Date', 'Workout Type', 'Calories Burned', 'Distance Travelled (m)', 
                'Steps Taken', 'Workout Length (minutes)', 
                'Breakfast', 'Lunch', 'Dinner', 
                'Calories Consumed', 'Net Calories'
            ]
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(filepath, index=False)
            return empty_df
        else:
            # Load data and ensure Date column is correct
            df = pd.read_csv(filepath)
            if 'Date' not in df.columns:
                st.warning("Date column missing. Reinitializing headers.")
                columns = [
                    'Date', 'Workout Type', 'Calories Burned', 'Distance Travelled (m)', 
                    'Steps Taken', 'Workout Length (minutes)', 
                    'Breakfast', 'Lunch', 'Dinner', 
                    'Calories Consumed', 'Net Calories'
                ]
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(filepath, index=False)
                return empty_df
            
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert Date column
            return df
    else:
        # File does not exist, initialize new database
        st.warning("File not found. Initializing new database.")
        columns = [
            'Date', 'Workout Type', 'Calories Burned', 'Distance Travelled (m)', 
            'Steps Taken', 'Workout Length (minutes)', 
            'Breakfast', 'Lunch', 'Dinner', 
            'Calories Consumed', 'Net Calories'
        ]
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(filepath, index=False)
        return empty_df

# Function to update the database
def update_database(
    date_str, workout_type, calories_burned, distance, steps, workout_length, 
    breakfast, lunch, dinner, calories_consumed, filepath='/Users/arnavagrawal/Documents/arda-HSC24/data/main_data.csv'
):
    df = load_database(filepath)  # Load the database
    net_calories = calories_consumed - calories_burned  # Calculate net calories
    
    # Add or update a row in the DataFrame
    new_entry = {
        'Date': date_str,
        'Workout Type': workout_type,
        'Calories Burned': calories_burned,
        'Distance Travelled (m)': distance,
        'Steps Taken': steps,
        'Workout Length (minutes)': workout_length,
        'Breakfast': breakfast,
        'Lunch': lunch,
        'Dinner': dinner,
        'Calories Consumed': calories_consumed,
        'Net Calories': net_calories
    }
    
    # Check if the date already exists and replace it, otherwise append
    if date_str in df['Date'].astype(str).values:
        df.loc[df['Date'] == date_str, :] = pd.DataFrame([new_entry])
    else:
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    
    # Save to CSV with Date as a column, not as an index
    df.to_csv(filepath, index=False)
    return df