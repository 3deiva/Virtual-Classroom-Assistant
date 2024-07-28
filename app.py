import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Get the current date and time for file naming
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

# Read the CSV file
csv_file = f"Attendance/Attendance_{date}.csv"

# Check if the file exists
try:
    df = pd.read_csv(csv_file)
    # Display the dataframe without highlighting
    st.dataframe(df)
except FileNotFoundError:
    st.error(f"File {csv_file} not found.")
except Exception as e:
    st.error(f"An error occurred: {e}")
