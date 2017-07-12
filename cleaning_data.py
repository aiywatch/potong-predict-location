# ## Import Libraries & Data
import pandas as pd
import numpy as np
import math


def clean_data(path):
    
    dataset = pd.read_csv(path)
    data = dataset.copy()
    
    
    # ## Dealing with time
    data['timestamp'] = pd.to_datetime(dataset['timestamp'])
    
    def get_day_of_week(dt):
        return dt.weekday()
    def get_hour(dt):
        return dt.hour
    
    data['day_of_week'] = data['timestamp'].apply(get_day_of_week)
    data['hour'] = data['timestamp'].apply(get_hour)
    
    
    ## ## Add location zone
    #data['location_zone'] = (data['linear_ref'] * 10000).apply(math.floor)
    #data[data['location_zone'] > 10000] = 10000
    #data[data['location_zone'] < 0] = 0
    
    
    # ## Duplicate data for training/testing
    
    # # 3nd previous point
    temp_data3 = data.copy()
    temp_data3['last_point_time'] = temp_data3['timestamp'].shift(3)
    temp_data3['second_from_last_point'] = (temp_data3['timestamp'] - temp_data3['timestamp'].shift(3)).dt.total_seconds()
    temp_data3['distance_from_last_point'] = temp_data3['linear_ref'] - temp_data3['linear_ref'].shift(3)
    temp_data3['last_point_location'] = temp_data3['linear_ref'].shift(3)
    temp_data3['last_point_lat'] = temp_data3['lat'].shift(3)
    temp_data3['last_point_lon'] = temp_data3['lon'].shift(3)
    temp_data3['last_point_speed'] = temp_data3['speed'].shift(3)
    
    # # 2nd previous point
    temp_data = data.copy()
    temp_data['last_point_time'] = temp_data['timestamp'].shift(2)
    temp_data['second_from_last_point'] = (temp_data['timestamp'] - temp_data['timestamp'].shift(2)).dt.total_seconds()
    temp_data['distance_from_last_point'] = temp_data['linear_ref'] - temp_data['linear_ref'].shift(2)
    temp_data['last_point_location'] = temp_data['linear_ref'].shift(2)
    temp_data['last_point_lat'] = temp_data['lat'].shift(2)
    temp_data['last_point_lon'] = temp_data['lon'].shift(2)
    temp_data['last_point_speed'] = temp_data['speed'].shift(2)
    
    
    # # Last point
    data['last_point_time'] = data['timestamp'].shift()
    data['second_from_last_point'] = (data['timestamp'] - data['timestamp'].shift()).dt.total_seconds()
    data['distance_from_last_point'] = data['linear_ref'] - data['linear_ref'].shift()
    data['last_point_location'] = data['linear_ref'].shift()
    data['last_point_lat'] = data['lat'].shift()
    data['last_point_lon'] = data['lon'].shift()
    data['last_point_speed'] = data['speed'].shift()
    
    
    # ## append all the data
    data = pd.concat([data, temp_data, temp_data3])
    
    
    
    # ## Remove Missing data and Outlier
    data.dropna(inplace=True)
    data = data.drop(data[data['distance_from_last_point'] <= 0].index)
    data = data.drop(data[data['distance_from_last_point'] > 0.2].index)
    data = data.drop(data[data['second_from_last_point'] > 1000].index)
    
    
    # ## Add location zone
    data['last_point_location_zone'] = (data['last_point_location'] * 10000).apply(math.floor)
    data[data['last_point_location_zone'] > 10000] = 10000
    data[data['last_point_location_zone'] < 0] = 0
    
    return data


def get_X_y(path):
    data = clean_data(path)
    # ## Filter relavant data
    data = data[['linear_ref', 'direction', 'day_of_week', 'hour', 'last_point_speed',
                 'last_point_location', 'second_from_last_point']]
    #data = data[['lat', 'lon', 'direction', 'day_of_week', 'hour', 'speed',
    #             'second_from_last_point',  'last_point_lat', 'last_point_lon']]
    
    #data = data[['distance_from_last_point', 'direction', 'day_of_week', 'hour', 'speed',
    #             'second_from_last_point', 'last_point_location']]
    
    
    # ## Split Features and Label
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return X, y
    