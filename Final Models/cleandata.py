import pandas as pd

#Load the data
file_path = "snwdata.csv"
data = pd.read_csv(file_path)

print("First few rows of the dataset:")
print(data.head())  #View the first few rows of the dataset

print("\nSummary of the dataset:")
print(data.info())  #Get a summary of the dataset

#Convert date and time columns to datetime format
data['Date'] = pd.to_datetime(data['Date'])
data['StartHour'] = data['StartHour'].replace("24:00:00", "00:00:00")
data['EndHour'] = data['EndHour'].replace("24:00:00", "00:00:00")

# Convert the StartHour and EndHour columns
data['StartHour'] = pd.to_datetime(data['StartHour'], format='%H:%M:%S').dt.time
data['EndHour'] = pd.to_datetime(data['EndHour'], format='%H:%M:%S').dt.time

#Handle missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())  #Checking for missing values

data = data.dropna()

#Extract useful features
data['Hour'] = data['Date'].dt.hour  
data['DayOfWeek'] = data['Date'].dt.day_name()
data['Month'] = data['Date'].dt.month_name()

#Normalize production data (optional)
#Normalizing the production values (0-1 scale)
data['NormalizedProduction'] = (data['Production'] - data['Production'].min()) / (data['Production'].max() - data['Production'].min())

#Inspect the cleaned and prepared data
print("\nCleaned and prepared data:")
print(data.head())

#Save the cleaned data (optional)
cleaned_file_path = "cleaned_snwdata.csv"
data.to_csv(cleaned_file_path, index=False)
