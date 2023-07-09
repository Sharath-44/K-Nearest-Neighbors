import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the saved KNN model
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

new_data = pd.read_csv('test_wo.csv')

# Perform label encoding on the new data
for col, le in label_encoders.items():
    if col in new_data.columns:
        new_data[col] = new_data[col].map(lambda x: le.transform([x])[
                                          0] if x in le.classes_ else -1)

y_pred = knn_model.predict(new_data)

actual_values = pd.read_csv('test.csv')
y_true = actual_values['remaining_distance_km']

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')

plt.xlabel('Data Point')
plt.ylabel('Remaining Distance (km)')
plt.title('Actual vs Predicted Remaining Distance')

plt.legend()

plt.show()
