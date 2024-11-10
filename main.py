import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv("data/exercise_data.csv")

df['date_time'] = pd.to_datetime(df['date_time'])

df = df.sort_values("date_time", ascending=True)


rng = np.random.default_rng()

# FRONTEND BEGINS HERE

st.write("""
# Welcome to ARDA!
### Hope your day is going great so far. Stay healthy!
         
""")

st.write("## New workout? Add it!")

col1, col2, col3 = st.columns(3)

with col1:
    workout_type = st.selectbox("Workout type", ("walking","running"))
    distance_moved = st.number_input("Distance travelled (m)", min_value=0, step=100)
with col2:
    calories_burned = st.number_input("Calories burned", min_value=0, step=50)
    workout_duration = st.number_input("Workout length (in minutes)", min_value=0, step=5)
with col3:
    steps = st.number_input("Steps taken", min_value=0, step=500)
    date = st.date_input("Date")
    time = st.time_input("Time")

if st.button("Record data"):
    st.write("Data recorded!")
    combined_date_time = datetime.datetime.combine(date, time)
    new_data = {
        'workout type': workout_type, 
        'distance_metres': distance_moved, 
        'cal_burned': calories_burned,
        'duration': workout_duration,
        'date_time': combined_date_time,
        'steps': steps
    }
    df.loc[len(df)] = new_data
    print(df)
    df.to_csv("data/exercise_data.csv", index=False)
    
if (st.button("Delete previous entry", type="primary")):
    if df.shape[0] == 0:
        st.write("No entries to delete!")
    else:
        df = df[:-1]
        df.to_csv("data/exercise_data.csv")
        st.write("Last entry deleted!")

st.write("## Your Data")
option = st.selectbox(
    'Which data would you like to see?',
    ('All Data', 'Past Week', 'Past Month')
)

if option == 'All Data':
    plt.bar(df["date_time"], df["steps"])
    plt.xticks(df["date_time"][0::30])
    st.pyplot(plt.gcf())
elif option == 'Past Week':
    
    past_week = df[df["date_time"] > datetime.datetime.now() - pd.to_timedelta("7day")]
    plt.bar(past_week["date_time"], past_week["steps"], width = 0.01)
    plt.xticks(past_week["date_time"][0::2])
    st.pyplot(plt.gcf())
elif option == 'Past Month':
    past_month = df[df["date_time"] > datetime.datetime.now() - pd.to_timedelta("30day")]
    plt.bar(past_month["date_time"], past_month["steps"], width= 0.1)
    plt.xticks(past_month["date_time"][0::3])
    st.pyplot(plt.gcf())