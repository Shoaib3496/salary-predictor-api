import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os

os.makedirs("models", exist_ok=True)

data = {
    "YearsExperience": [0.5, 1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2,
                        3.7, 3.9, 4.0, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.8],
    "Salary": [39343, 46205, 37731, 43525, 39891, 45000, 56642, 60150, 54445, 64445,
               57189, 63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363]
}
df = pd.DataFrame(data)
X = df[["YearsExperience"]]
y = df["Salary"]

lr_model = LinearRegression().fit(X, y)
with open("models/model_v1.pkl", "wb") as f:
    pickle.dump(lr_model, f)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
with open("models/model_v2.pkl", "wb") as f:
    pickle.dump(rf_model, f)