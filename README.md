📈 Stock Price Prediction App
A web application that predicts future stock prices using an LSTM model. Built with Flask, TensorFlow, yfinance, and Plotly, this app provides 10-day forecasts of stock closing prices with a clean and modern user interface.

🚀 Features
📉 Predicts future stock closing prices (e.g., for TSLA)

🤖 Uses a trained LSTM neural network for time series prediction

🎨 Stylish, modern frontend UI with Plotly charts

📦 Flask-powered backend with real-time yfinance data

🧠 Scaled data preprocessing using MinMaxScaler

🛠 Tech Stack
Tool / Library	Purpose
Flask	Web framework
TensorFlow / Keras	LSTM model for predictions
yfinance	Real-time stock data
joblib	Save/load scaler
Plotly	Interactive charts
HTML/CSS	Frontend UI
📂 Project Structure

├── app.py               # Flask web server
├── train_stock.py       # Script to train and save LSTM model
├── templates/
│   └── index.html       # HTML template with modern UI and Plotly integration
├── stock_model.h5       # Trained LSTM model
├── scaler_stock.pkl     # Scaler used for price normalization
├── requirements.txt     # Required Python packages
└── README.md            # This file
🧪 1. Train the Model
Before running the app, train the model with historical stock data:

python train_stock.py
This will create:

stock_model.h5 — the trained LSTM model

scaler_stock.pkl — the fitted scaler used for normalizing data

🌐 2. Run the Web App
Start the Flask server:

python app.py
Then open your browser and go to:

http://127.0.0.1:5000/

📦 Example Prediction Output
Input: Symbol = TSLA, Days = 10

Output: 10-day closing price forecast + interactive chart
![image](https://github.com/user-attachments/assets/c2e27a3e-278c-4c4b-8313-4c4f0a58232d)
![image](https://github.com/user-attachments/assets/1a8aa423-9b6e-4388-af73-6fbc581d25c6)

📌 Notes
Minimum of 60 days of historical data required for predictions.

If yfinance returns empty data, the app will prompt with an error.

To test another stock symbol, input any valid ticker (e.g., AAPL, GOOGL, etc.)

📬 
For improvements or issues, feel free to open an issue or PR.
