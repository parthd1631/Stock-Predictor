from flask import Flask, render_template, jsonify, request
from models.predictor import StockPredictor
import os
import traceback
import logging

app = Flask(__name__)
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

predictor = StockPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '')
        days = data.get('days', 30)  # Default to 30 days if not specified
        
        if not ticker:
            return jsonify({'error': 'No ticker provided'}), 400
        
        # Normalize ticker to uppercase
        ticker = ticker.upper().strip()
        
        # Validate days parameter
        try:
            days = int(days)
            if days < 1 or days > 365:
                days = 30  # Reset to default if out of reasonable range
        except (ValueError, TypeError):
            days = 30  # Reset to default if not a valid integer
        
        logger.info(f"Processing prediction request for ticker: {ticker}, days: {days}")
        
        # Make prediction
        result = predictor.predict_stock(ticker, days)
        
        if 'error' in result:
            logger.error(f"Error in prediction: {result['error']}")
            return jsonify({'error': result['error']}), 500
        
        logger.info(f"Prediction successful for {ticker}: {result['prediction']}")
        return jsonify(result)
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Unhandled exception in predict endpoint: {str(e)}\n{error_details}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 