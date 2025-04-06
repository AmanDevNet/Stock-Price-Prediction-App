from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objs as go
import datetime

app = Flask(__name__)

# Load model and scaler
model = load_model("stock_model.h5")
scaler = joblib.load("scaler_stock.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    predicted_prices = []
    last_price = None
    stock_symbol = ""
    days = 0
    error = None
    graph_json = None

    if request.method == "POST":
        stock_symbol = request.form.get("symbol", "").upper()
        days = request.form.get("days", "")

        if not stock_symbol or not days.isdigit():
            error = "Please enter a valid stock symbol and number of days."
            return render_template("index.html", error=error)

        days = int(days)

        try:
            # Load recent data
            end_date = datetime.datetime.today()
            start_date = end_date - datetime.timedelta(days=100)
            df = yf.download(stock_symbol, start=start_date, end=end_date)

            if df.empty or len(df) < 60:
                error = "Not enough data to make a prediction."
                return render_template("index.html", error=error)

            close_prices = df["Close"].values.reshape(-1, 1)
            scaled_data = scaler.transform(close_prices)

            X_input = scaled_data[-60:]
            predictions = []

            for _ in range(days):
                input_reshaped = np.reshape(X_input, (1, X_input.shape[0], 1))
                pred_scaled = model.predict(input_reshaped, verbose=0)
                predictions.append(pred_scaled[0][0])
                X_input = np.append(X_input, pred_scaled)[1:]

            predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            predicted_prices = [f"{p:.2f}" for p in predicted_prices]  # For display
            predicted_numeric = [float(p) for p in predicted_prices]  # For Plotly
            last_price = f"{close_prices[-1][0]:.2f}"

            # Create interactive plot
            dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(days + 1)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=[float(last_price)] + predicted_numeric, 
                                   mode='lines+markers', name='Predicted Price'))
            fig.update_layout(title=f'{stock_symbol} Stock Price Prediction', xaxis_title='Date', 
                            yaxis_title='Price (USD)', template='plotly', height=400)
            graph_json = fig.to_json()

        except Exception as e:
            error = str(e)

    return render_template("index.html", predicted_prices=predicted_prices, last_price=last_price,
                          stock_symbol=stock_symbol, days=days, error=error, graph_json=graph_json)

if __name__ == "__main__":
    app.run(debug=True)