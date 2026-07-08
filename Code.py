import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Telco-Customer-Churn.csv')
print(df.shape)
print(df.info())
print(df.head())

#missing_pct = (df.isna().sum() / len(df) * 100).round(2)
#print(missing_pct)

df['Churn'].value_counts(normalize=True)
print(df[df['TotalCharges'] == ' '][['tenure','TotalCharges']])
