import streamlit as st

distance_moved = 15000
total = 99999

st.progress(distance_moved/total, text=f"Progress towards target: {round(distance_moved * 100/total, 1)}%")