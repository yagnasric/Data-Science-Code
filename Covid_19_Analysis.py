import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

#Importing Datasets

df_clean = pd.read_csv('covid_19_clean_complete.csv')
df_country = pd.read_csv('country_wise_latest.csv')
df_day = pd.read_csv('day_wise.csv')
df_usa = pd.read_csv('usa_county_wise.csv')
dF_world = pd.read_csv('worldometer_data.csv')
df_grouped = pd.read_csv('full_grouped.csv')

#Checkong Null Values

# print('Clean Comaplete', df_clean.isna().sum())
# print('Country Wise', df_country.isna().sum())
# print('Day Wise', df_day.isna().sum())
# print('USA Wise', df_usa.isna().sum())
# print('World Meter', dF_world.isna().sum())
# print('Gorup Wise', df_grouped.isna().sum())

df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
df_clean.fillna(0, inplace= True)

global_ts = df_clean.groupby('Date')[['Confirmed', 'Deaths', 'Recovered']].sum()

print(global_ts)

plt.figure(figsize=(14,6))
plt.plot(global_ts.index, global_ts['Confirmed'], label='Confirmed')
plt.plot(global_ts.index, global_ts['Deaths'], label='Deaths')
plt.plot(global_ts.index, global_ts['Recovered'], label='Recovered')
plt.show()

latest = df_clean[df_clean['Date'] == df_clean['Date'].max()]

country_totals = latest.groupby('Country/Region')[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

top15 = country_totals.sort_values('Confirmed', ascending= False).head(15)


plt.figure(figsize=(12,6))
sns.barplot(y= 'Country/Region', x = 'Confirmed', data = top15)
plt.show()

top6 = country_totals.sort_values('Confirmed', ascending= False).head(6)['Country/Region']

plt.figure(figsize=(14,7 ))

for c in top6:
    sub = df_clean[df_clean['Country/Region'] == c].groupby('Date')['Confirmed'].sum()
    plt.plot(sub.index, sub.values, label= c)

plt.show()

daily_new = global_ts.diff().fillna(global_ts)

plt.figure(figsize=(14, 5))
plt.bar(daily_new.index, daily_new['Confirmed'], color='orange')
plt.show()

country_totals['cfr'] = country_totals['Deaths']/country_totals['Confirmed']

top_cfr = country_totals[country_totals['Confirmed'] > 1000].sort_values('cfr', ascending=False).head(15)


plt.figure(figsize=(12, 6))
sns.barplot(y = 'cfr', x = 'Country/Region', data = top_cfr)
plt.show()


corr = country_totals[['Confirmed', 'Deaths', 'Recovered']].corr()

sns.heatmap(corr, annot = True, cmap = 'coolwarm', center = 0 )
plt.show()


df_country['Recovery Rate'] = (df_country['Recovered']/df_country['Confirmed'])*100


top_recovery = df_country.sort_values('Recovery Rate', ascending= False).head(10)
print(top_recovery)

plt.figure(figsize=(10, 9))
sns.barplot(x = 'Recovery Rate', y = 'Country/Region', data = top_recovery)
plt.show()


state_cases = df_usa.groupby('Province_State') ['Confirmed'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
state_cases.plot(kind="barh", color="orange")
plt.show()


top_tests = dF_world.sort_values('TotalTests', ascending= False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x = 'TotalTests', y ='Country/Region', data = top_tests)
plt.show()


top_tests = df_world.sort_values("TotalTests", ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=top_tests, x="TotalTests", y="Country/Region", palette="Blues_r")
plt.title("Top 10 Countries by Total Tests")
plt.show()

df_world["TestsPerMillion"] = df_world["TotalTests"] / (df_world["Population"]/1e6)
top_per_million = df_world.sort_values("TestsPerMillion", ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=top_per_million, x="TestsPerMillion", y="Country/Region", palette="Purples")
plt.title("Countries with Most Tests per Million People")
plt.show()










