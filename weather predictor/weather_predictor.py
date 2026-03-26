import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
df = pd.read_csv('DailyDelhiClimate.csv')
df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear
df['month'] = df['date'].dt.month
df = df.ffill()
df['target_temp'] = df['meantemp'].shift(-1)
df = df.dropna()
features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure', 'day_of_year', 'month']
X = df[features]
y = df['target_temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = MLPRegressor(hidden_layer_sizes=(100, 50),max_iter=200,learning_rate_init=0.01,random_state=42,verbose=False)
print("Training the model...")
model.fit(X_train, y_train)
print("Training completed.")
predictions = model.predict(X_test)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(model.loss_curve_, color='crimson', linewidth=2)
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(y_test.values[:50], label='Actual Temperature', color='royalblue', linewidth=2, marker='o', markersize=4)
plt.plot(predictions[:50], label='Predicted Temperature', color='darkorange', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Temperature (First 50 days)')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
mae = mean_absolute_error(y_test, predictions)
print(f"\nMean Absolute Error: {mae:.2f}°C")
