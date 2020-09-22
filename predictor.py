import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression

data = pd.read_csv("ipl.csv")

data.columns = ['mid', 'date', 'venue', 'bat_team', 'bowl_team', 'batsman', 'bowler',
       'runs', 'wickets', 'overs', 'runs_last_6', 'wickets_last_6', 'striker',
       'non-striker', 'total']

data = data.drop(['batsman','bowler','mid','striker','non-striker'],axis=1).copy()

# considering the only teams which are Playing the IPL in Current seasons

consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
#removing inconsistent teams and selecting only consistent_teams from  the dataset
# Keeping only consistent teams
print('Before removing inconsistent teams: {}'.format(data.shape))
data = data[(data['bat_team'].isin(consistent_teams)) & (data['bowl_team'].isin(consistent_teams))]
print('After removing inconsistent teams: {}'.format(data.shape))


# Removing the first 6 overs data in every match
print('Before removing first 6 overs data: {}'.format(data.shape))
data = data[data['overs']>=6.0]
print('After removing first 6 overs data: {}'.format(data.shape))


# Converting the date Column form string into Data time object
print("Before converting date column from string to datetime object: {}".format(type(data.iloc[0,0])))

# Converting the column 'date' from string into datetime object

print("Before converting 'date' column from string to datetime object: {}".format(type(data.iloc[0,0])))
data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
print("After converting 'date' column from string to datetime object: {}".format(type(data.iloc[0,0])))

# Data Preprocessing
encoded_data = pd.get_dummies(data = data, columns= ["bat_team",'bowl_team'])
ncoded_data= encoded_data[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_6', 'wickets_last_6', 'total']]


#Splitting the dataset into Training data and Test data

#dt can be used to access the values of the series as datetimelike and return several properties. Pandas Series. dt. minute attribute return a numpy array containing the minutes of the datetime in the underlying data of the given series object.

X_train = ncoded_data.drop("total",axis=1)[ncoded_data['date'].dt.year <=2016]
X_test = ncoded_data.drop("total",axis =1)[ncoded_data['date'].dt.year >= 2017]
X_val = ncoded_data.drop("total",axis=1)[ncoded_data['date'].dt.year >=2017]

Y_train = ncoded_data[encoded_data['date'].dt.year <= 2016]['total'].values
Y_test = ncoded_data[encoded_data['date'].dt.year >= 2017]['total'].values
Y_val = ncoded_data[encoded_data['date'].dt.year >= 2017]['total'].values

X_train = X_train.drop("date",axis=1)
X_test = X_test.drop("date",axis = 1)
X_val = X_val.drop("date",axis=1)


linear_regressor = LinearRegression()
linear_regressor.fit(X_train,Y_train)

#creating a pickle file for  the classifier
filename = 'Ipl_first_innings_score_prediction_model.pkl'
pickle.dump(linear_regressor,open(filename,'wb'))
