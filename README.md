# delhi-weather-predictor
Delhi Daily Temperature Prediction 🌡️
An AI-powered regression model designed to predict the next day's mean temperature in Delhi using historical climate data and Neural Networks.

📌 Project Overview
This project utilizes the MLPRegressor (Multi-Layer Perceptron) from Scikit-Learn to perform time-series forecasting. By analyzing features like humidity, wind speed, and barometric pressure, the model learns the seasonal patterns of Delhi’s climate to predict future temperature fluctuations.

🛠️ Tech Stack
Language: Python

Data Manipulation: pandas, numpy

Machine Learning: scikit-learn

Visualization: matplotlib

📊 Methodology
1. Feature Engineering
The raw dataset is transformed to make it "time-aware" for the Neural Network:

Temporal Features: Extracted day_of_year and month to capture seasonality.

Lagged Target: Created a target_temp by shifting the meantemp by -1 day, allowing the model to learn the relationship between today's conditions and tomorrow's heat.

Data Cleaning: Forward-filling (ffill) missing values to maintain time continuity.

2. The Neural Network Architecture
The model uses a feed-forward Artificial Neural Network (ANN) with the following configuration:

Hidden Layers: Two layers with 100 and 50 neurons respectively.

Optimizer: Adam (default in MLPRegressor).

Learning Rate: Initialized at 0.01 for faster initial convergence.

🚀 How to Run
Ensure you have the DailyDelhiClimate.csv file in your root directory.

Install dependencies:

Bash
pip install pandas scikit-learn matplotlib numpy
Execute the script:

Bash
python weather_predict.py
📈 Results & Visualizations
The script generates two primary plots to evaluate performance:

Training Heartbeat: A loss curve showing how the model's error decreases over training epochs.

Actual vs. Predicted: A side-by-side comparison of the real temperatures versus the AI's predictions for the test period.

Note: The final Mean Absolute Error (MAE) typically hovers around 1.5°C to 2.5°C, depending on the dataset's variance.
