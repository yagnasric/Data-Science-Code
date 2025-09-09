import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

pd.set_option('display.max_columns', None)

# Some basic EDA analysis on the Uber Dataset

df  = pd.read_csv('ncr_ride_bookings.csv')

df['Vehicle Type'] = df['Vehicle Type'].replace({'eBike : E-Bike'})

print(df['Vehicle Type'])

vtal_mean = np.nanmean(df['Avg VTAT'])

df['Avg VTAT'] = df['Avg VTAT'].fillna(vtal_mean)

ctat_mean = np.nanmean(df['Avg CTAT'])

df['Avg CTAT'] = df['Avg CTAT'].fillna(ctat_mean)

df['Reason for cancelling by Customer']= df['Reason for cancelling by Customer'].fillna('Unknown')
df['Driver Cancellation Reason'] = df['Driver Cancellation Reason'].fillna('Unknown') 
df[ 'Incomplete Rides Reason'] = df[ 'Incomplete Rides Reason'].fillna('Unknown')

print(df['Booking Status'].unique())

total_cancelled = df['Booking Status'].isin(['Cancelled by Driver','Cancelled by Customer']).sum()
total_rides = len(df['Booking Status'])

#Calculate percentage of cancelled rides
cancelled_percentage = (total_cancelled/total_rides)*100

print(cancelled_percentage)

print(df['Booking Status'].value_counts())


# Random Forest Classifier on Uber Dataset to predict the Booking Status 

df1 = df.drop(["Booking ID","Customer ID","Reason for cancelling by Customer","Driver Cancellation Reason","Incomplete Rides Reason","Cancelled Rides by Customer","Cancelled Rides by Driver","Incomplete Rides", 'Vehicle Type', 'Pickup Location', 'Drop Location', 'Payment Method'],axis=1)

df1["Date"]=pd.to_datetime(df["Date"])
df1["Time"]=pd.to_datetime(df1["Time"],format="mixed").dt.hour
df1["Day of Week"]=df1["Date"].dt.dayofweek
df1["Month"]=df1["Date"].dt.month
df1["Year"]=df1["Date"].dt.year

df1 = df1.drop(['Date'], axis=1)

for col in ["Avg VTAT","Avg CTAT","Booking Value","Ride Distance","Driver Ratings","Customer Rating"]:
    df[col]=df[col].fillna(df[col].mean())
    
    
for col in ["Payment Method"]:
    df[col]=df[col].fillna(df[col].mode()[0])
    
categorical=["Booking Status","Vehicle Type","Pickup Location","Drop Location","Payment Method"]
for col in categorical:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    
y=df1["Booking Status"]
X=df1.drop(["Booking Status"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rfc = RandomForestClassifier(n_estimators=200, random_state=42)
model = rfc.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print(ac)
print(cm)
