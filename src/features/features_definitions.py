import pandas as pd
import numpy as np
#import plotly.express as px
import seaborn as sns
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
import pathlib
warnings.filterwarnings('ignore') 

def filter_dataframe(df):
    # Define the values to be filtered out
    exclude_pick_ups = ['secsec', 'Railway StRailway St', 'sectsect', 'pp']
    exclude_cab_type = ['UberGoPerson2']
    
    # Filter the DataFrame for 'Pick_Up'
    df = df[~df['Pick_Up'].isin(exclude_pick_ups)]
    
    # Filter the DataFrame for 'Cab_Type'
    df = df[df['Cab_Type'] != exclude_cab_type[0]]

    return df

def map_cab_type(df):
    mapping = {
        'UberGoPerson1': 'Hatchbacks',
        'PremierPerson1': 'Sedan',
        'UberAutoPerson1': 'Auto',
        'MOTOPerson1': 'Bike',
        'UberXLPerson1': 'SUV'
    }
    
    # Replace the 'Cab_Type' values according to the mapping
    df['Cab_Type'] = df['Cab_Type'].replace(mapping)
    return df

def clean_cab_price(df):
    df['Cab_Price'] = df['Cab_Price'].str.replace('₹', '').str.replace(',', '').astype(float)
    return df


def split_arrival_time(df):
    # Split the 'Arrival_Time' column into two new columns
    split_columns = df['Arrival_Time'].str.split('•', expand=True)
    df['Cab_Arrival_Time'] = split_columns[0]
    df['Cab_Destination_Time'] = split_columns[1]
    
    # Remove the original 'Arrival_Time' column from the DataFrame
    del df['Arrival_Time']
    return df


def split_current_time(df):
    # Split the 'Current Time' column into two new columns
    df[['Current_Date', 'Current_Time']] = df['Current Time'].str.split(',', n=1, expand=True)

    # Drop the original 'Current Time' column from the DataFrame
    df.drop(columns=['Current Time'], inplace=True)
    return df


def apply_availability_mapping(df):
    def map_availability(arrival_time):
        return 0 if arrival_time == 'Unavailable' else 1

    df['Availability'] = df['Cab_Arrival_Time'].apply(map_availability)
    return df

def clean_cab_destination_time(df):
    df['Cab_Destination_Time'] = df['Cab_Destination_Time'].str.replace(' drop-off', '', regex=False)
    return df

def format_current_time(df):
    df['Current_Time'] = df['Current_Time'].str.split(':').str[:2].str.join(':')
    return df

def update_destination_time_for_unavailability(df):
    df.loc[df['Availability'] == 0, 'Cab_Destination_Time'] = '0'
    return df

def calculate_route_time(df):
    df['Route_Time'] = (pd.to_datetime(df['Cab_Destination_Time'].str.strip(), errors='coerce') 
                        - pd.to_datetime(df['Current_Time'].str.strip(), errors='coerce')).dt.total_seconds() / 60
    return df

def split_current_time_into_hour_minute(df):
    df[['Current_Hour', 'Current_Minute']] = df['Current_Time'].str.split(':', expand=True)
    return df

def split_current_date(df):
    df[['Current_Date', 'Current_Month', 'Current_Year']] = df['Current_Date'].str.split('/', expand=True)
    return df

def convert_time_columns_to_int(df):
    df['Current_Hour'] = df['Current_Hour'].astype('int')
    df['Current_Minute'] = df['Current_Minute'].astype('int')
    df['Current_Month'] = df['Current_Month'].astype('int')
    df['Current_Year'] = df['Current_Year'].astype('int')
    return df


def test_feature_build(df):
    filter_dataframe(df)
    map_cab_type(df)
    clean_cab_price(df)
    split_arrival_time(df)
    split_current_time(df)
    apply_availability_mapping(df)
    clean_cab_destination_time(df)
    format_current_time(df)
    update_destination_time_for_unavailability(df)
    calculate_route_time(df)
    split_current_time_into_hour_minute(df)
    split_current_date(df)
    convert_time_columns_to_int(df)
    print(df.head())

def feature_build(df, tag):
    filter_dataframe(df)
    map_cab_type(df)
    clean_cab_price(df)
    split_arrival_time(df)
    split_current_time(df)
    apply_availability_mapping(df)
    clean_cab_destination_time(df)
    format_current_time(df)
    update_destination_time_for_unavailability(df)
    calculate_route_time(df)
    split_current_time_into_hour_minute(df)
    split_current_date(df)
    convert_time_columns_to_int(df)
    
    feature_names = [f for f in df.columns]
    print(f'We have {len(feature_names)} features in {tag}.')
    return df[feature_names]



if __name__ == "__main__":
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_path = home_dir.as_posix() + '/data/raw/Cab_Data.csv'
    data = pd.read_csv(data_path, nrows=10)
    test_feature_build(data)