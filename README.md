# Stock Price Predictor

A full-stack web application that predicts stock price movements using machine learning.

## Features

- Enter any stock symbol to predict tomorrow's price movement
- Uses historical stock data from Yahoo Finance
- Machine learning model with 60% prediction accuracy
- Easy-to-use web interface

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/Stock-Predictor.git
cd Stock-Predictor
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Running the Application

### Option 1: Using the run script (recommended)

```
python run.py
```
or
```
./run.py
```

### Option 2: Running directly with Flask

```
cd app
python app.py
```

Then, open a web browser and navigate to:
```
http://127.0.0.1:8080/
```

## How It Works

The application uses a Random Forest Classifier trained on historical stock data to predict whether a stock's price will go up or down the next day. Features include:

- Price ratio to moving averages (2, 5, 60, and 250 days)
- Price trends over multiple timeframes

## Disclaimer

This application is for educational purposes only. The predictions should not be used as financial advice. Always do thorough research before making investment decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.