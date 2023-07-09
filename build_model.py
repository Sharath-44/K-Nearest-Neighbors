import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('final.csv')

X = data.drop('remaining_distance_km', axis=1)
y = data['remaining_distance_km']

categorical_cols = ['thread_pattern', 'road_conditions', 'driving_style',
                    'maintenance_history', 'driving_environment', 'tire_composition', 'tire_size']

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

y_pred = knn_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Remaining Distance (km)')
plt.ylabel('Predicted Remaining Distance (km)')
plt.title('Actual vs Predicted Remaining Distance')
plt.show()
