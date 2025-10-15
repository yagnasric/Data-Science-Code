import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
                                                           
service_weekly = pd.read_csv('services_weekly.csv')
staff = pd.read_csv('staff.csv')
patients = pd.read_csv('patients.csv')
staff_schedule = pd.read_csv('staff_schedule.csv')

print('Columns of services weely are:', services_weekly.columns.tolist())

plt.figure(figsize=(8, 5))

ax=sns.barplot(data=patients, x="service", y="satisfaction", estimator="mean", errorbar=None, palette="viridis")

for containers in ax.containers:
    ax.bar_label(containers)
    
plt.show()

bins = [0, 10, 20, 30, 40, 50, 60, 150]
labels = ["0–10", "10–20", "20–30", "30–40", "40–50", "50–60", "60+"]

patients['age_group'] = pd.cut(patients['age'], bins=bins, labels=labels, right=False)

age_mean = (
      patients.groupby('age_group', as_index = False, observed= True)['satisfaction'].mean()
    )

plt.figure(figsize=(8, 5))

ax = sns.barplot(data = age_mean, x = 'age_group', y = 'satisfaction', palette='viridis')

for containers in ax.containers:
    ax.bar_label(containers)
    
plt.show()

patients['arrival_month'] = patients['arrival_date'].str[5:7].astype(int)

plt.figure(figsize=(12, 5))

ax= sns.barplot(data=patients, x='arrival_month', y = 'satisfaction', estimator='mean',  palette = 'viridis')

for containers in ax.containers:
    ax.bar_label(containers)
    
plt.show()

cols_to_plot = ['available_beds', 'patients_request', 'patients_admitted',
                'patients_refused', 'staff_morale', 'event', 'patient_satisfaction']

# Pairplot
sns.pairplot(service_weekly[cols_to_plot], y_vars=['patient_satisfaction'], x_vars=cols_to_plot, height=4, aspect=1)
plt.suptitle("Variables vs Patient Satisfaction", y=1.02)
plt.show()