import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import datetime


#Header
st.header("Sustainable Planet Dashbaord")

csv_file = st.file_uploader("upload file", type={"csv", "txt"})
if csv_file is not None:
    data = pd.read_csv(csv_file) #path folder of the data file
    #real data from CSV
    df = pd.DataFrame(data) #data into dataframe

    #remove rows with missing dates keep rows with dates
    df = df[df['Reporting Year End Date'].notna()]

    df['Reporting Year End Date'] = pd.to_datetime(df['Reporting Year End Date'], format='%d/%m/%y', errors='coerce') #convert coloumn type to date and remove dates that do not follow date format using error (19 anomolies)
    print(df.dtypes) #print datatypes so end date can be displayes as datetime64 instead of object
    
    st.dataframe(df) #display clean data
    st.line_chart(data = df, x='Client Name', y='Total Footprint (tCO2e)') #create line chart with data
    print(df.index)

    df.sort_values(by='Reporting Year End Date', inplace = True) #sort dataframe by date
    st.dataframe(df) #display clean data in date order
    print(df.index)
    st.line_chart(data = df, x='Reporting Year End Date', y='Total Footprint (tCO2e)') #create line chart with data but for dates over time

