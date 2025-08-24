import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
data = pd.read_csv('Admission_Prediction.csv')
print(data.describe())
print(data.isnull().sum())
