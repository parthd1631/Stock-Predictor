import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
from datetime import datetime, timedelta
import logging
import traceback

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
        self.predictors = []
        self.horizons = [2, 5, 60, 250]
        
    def prepare_data(self, ticker):
        try:
            logger.info(f"Downloading data for {ticker}")
            # Download stock data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1500)
            
            # Get stock data from Yahoo Finance
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            # Check if data was retrieved
            if stock_data.empty:
                logger.error(f"No data found for ticker {ticker}")
                raise ValueError(f"No data found for ticker {ticker}. Please check the symbol and try again.")
            
            logger.info(f"Downloaded {len(stock_data)} rows of data for {ticker}")
            
            # Create a copy to avoid pandas alignment issues
            df = stock_data.copy()
            
            # Add tomorrow's price column
            df['Tomorrow'] = df['Close'].shift(-1)
            
            # Create target - 1 if tomorrow's price > today's price, 0 otherwise
            # Using a safe element-wise approach
            tomorrow = df['Tomorrow'].values
            close = df['Close'].values
            target = np.zeros(len(df))
            for i in range(len(df) - 1):  # -1 because last row's Tomorrow is NaN
                if tomorrow[i] > close[i]:
                    target[i] = 1
            
            df['Target'] = target
            
            # Create predictors
            self.predictors = []
            
            for horizon in self.horizons:
                # Rolling average price
                rolling_averages = df.rolling(horizon).mean()
                
                # Ratio of price to rolling average
                ratio_column = f"Close_Ratio_{horizon}"
                df[ratio_column] = df['Close'] / rolling_averages['Close']
                
                # Trend indicator
                trend_column = f"Trend_{horizon}"
                df[trend_column] = df.shift(1).rolling(horizon).sum()['Target']
                
                self.predictors += [ratio_column, trend_column]
            
            # Drop NA values
            df = df.dropna()
            
            if len(df) < 100:
                logger.error(f"Insufficient data for {ticker} after preprocessing. Only {len(df)} valid rows.")
                raise ValueError(f"Insufficient historical data for {ticker}. Need at least 100 valid trading days.")
            
            return df
        except Exception as e:
            logger.error(f"Error preparing data for {ticker}: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def train(self, stock_data):
        try:
            logger.info(f"Training model on {len(stock_data)} rows of data")
            # Train the model using all available data
            X = stock_data[self.predictors]
            y = stock_data['Target']
            self.model.fit(X, y)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def predict(self, stock_data):
        try:
            # Make prediction for the most recent data point
            latest_data = stock_data[self.predictors].iloc[-1:].copy()
            logger.info(f"Making prediction using features: {latest_data.values}")
            
            prediction_prob = self.model.predict_proba(latest_data)[0, 1]
            prediction = 1 if prediction_prob >= 0.6 else 0
            
            logger.info(f"Prediction result: {prediction} with probability {prediction_prob}")
            
            # Format the date as a string to ensure it's JSON serializable
            date_str = stock_data.index[-1].strftime('%Y-%m-%d')
            
            return {
                'ticker_date': date_str,
                'probability': float(round(prediction_prob * 100, 2)),  # Convert to float
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': 'High' if abs(prediction_prob - 0.5) > 0.2 else 'Medium' if abs(prediction_prob - 0.5) > 0.1 else 'Low'
            }
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def calculate_technical_indicators(self, stock_data):
        """Calculate common technical indicators for the stock data."""
        try:
            # Use only the last 250 trading days for technical indicators
            recent_data = stock_data.tail(250).copy()
            
            # RSI (Relative Strength Index)
            delta = recent_data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            recent_data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema12 = recent_data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = recent_data['Close'].ewm(span=26, adjust=False).mean()
            recent_data['MACD'] = ema12 - ema26
            recent_data['MACD_Signal'] = recent_data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Moving Averages
            recent_data['MA_50'] = recent_data['Close'].rolling(window=50).mean()
            recent_data['MA_200'] = recent_data['Close'].rolling(window=200).mean()
            
            # Volume indicators
            recent_data['Volume_Change'] = recent_data['Volume'].pct_change()
            recent_data['Volume_MA'] = recent_data['Volume'].rolling(window=20).mean()
            
            # NEW: Bollinger Bands
            rolling_mean = recent_data['Close'].rolling(window=20).mean()
            rolling_std = recent_data['Close'].rolling(window=20).std()
            recent_data['BB_Upper'] = rolling_mean + (rolling_std * 2)
            recent_data['BB_Lower'] = rolling_mean - (rolling_std * 2)
            recent_data['BB_Width'] = (recent_data['BB_Upper'] - recent_data['BB_Lower']) / rolling_mean
            
            # NEW: Average True Range (ATR) for volatility
            high_low = recent_data['High'] - recent_data['Low']
            high_close = np.abs(recent_data['High'] - recent_data['Close'].shift())
            low_close = np.abs(recent_data['Low'] - recent_data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            recent_data['ATR'] = true_range.rolling(14).mean()
            
            # NEW: Money Flow Index (MFI)
            typical_price = (recent_data['High'] + recent_data['Low'] + recent_data['Close']) / 3
            money_flow = typical_price * recent_data['Volume']
            
            # Positive and negative money flow
            positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=14).sum()
            
            # Money ratio & money flow index
            money_ratio = positive_flow / negative_flow
            recent_data['MFI'] = 100 - (100 / (1 + money_ratio))
            
            # Get the most recent indicators
            last_row = recent_data.iloc[-1]
            
            # Determine signals
            rsi_value = float(last_row['RSI'])  # Convert Series to float
            macd_value = float(last_row['MACD'])
            macd_signal = float(last_row['MACD_Signal'])
            ma_50 = float(last_row['MA_50'])
            ma_200 = float(last_row['MA_200'])
            volume = float(last_row['Volume'])
            volume_ma = float(last_row['Volume_MA'])
            
            # NEW: Get new indicator values
            bb_upper = float(last_row['BB_Upper'])
            bb_lower = float(last_row['BB_Lower'])
            bb_width = float(last_row['BB_Width'])
            atr = float(last_row['ATR'])
            mfi = float(last_row['MFI'])
            
            # Current price
            current_price = float(last_row['Close'])
            
            # RSI signal
            if rsi_value < 30:
                rsi_signal = "Oversold"
                rsi_trend = "bullish"
            elif rsi_value > 70:
                rsi_signal = "Overbought"
                rsi_trend = "bearish"
            else:
                rsi_signal = "Neutral"
                rsi_trend = "neutral"
            
            # MACD signal
            if macd_value > macd_signal:
                macd_trend = "bullish"
                macd_signal_text = "Bullish Crossover"
            else:
                macd_trend = "bearish"
                macd_signal_text = "Bearish Crossover"
            
            # Moving Average signal
            if ma_50 > ma_200:
                ma_trend = "bullish"
                ma_signal = "Golden Cross"
            else:
                ma_trend = "bearish"
                ma_signal = "Death Cross"
            
            # Volume signal
            if volume > volume_ma:
                volume_trend = "bullish"
                volume_signal = "Above Average"
            else:
                volume_trend = "bearish"
                volume_signal = "Below Average"
                
            # NEW: Bollinger Bands signal
            if current_price > bb_upper:
                bb_signal = "Overbought"
                bb_trend = "bearish"
            elif current_price < bb_lower:
                bb_signal = "Oversold"
                bb_trend = "bullish"
            else:
                bb_signal = "Within Bands"
                bb_trend = "neutral"
                
            # NEW: MFI signal
            if mfi < 20:
                mfi_signal = "Oversold"
                mfi_trend = "bullish"
            elif mfi > 80:
                mfi_signal = "Overbought"
                mfi_trend = "bearish"
            else:
                mfi_signal = "Neutral"
                mfi_trend = "neutral"
            
            # NEW: Calculate risk metrics
            # Volatility (based on ATR as % of price)
            volatility_pct = (atr / current_price) * 100
            if volatility_pct < 1.5:
                volatility_level = "Low"
            elif volatility_pct < 3.0:
                volatility_level = "Medium"
            else:
                volatility_level = "High"
                
            # Calculate Sharpe ratio (simplified)
            returns = recent_data['Close'].pct_change().dropna()
            mean_return = returns.mean() * 252  # Annualized
            std_return = returns.std() * np.sqrt(252)  # Annualized
            risk_free_rate = 0.03  # Assume 3% risk-free rate
            sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
            
            return {
                'rsi': {
                    'value': round(rsi_value, 2),
                    'signal': rsi_signal,
                    'trend': rsi_trend
                },
                'macd': {
                    'value': round(macd_value, 2),
                    'signal': macd_signal_text,
                    'trend': macd_trend
                },
                'moving_average': {
                    'ma50': round(ma_50, 2),
                    'ma200': round(ma_200, 2),
                    'signal': ma_signal,
                    'trend': ma_trend
                },
                'volume': {
                    'value': int(volume),
                    'avg': int(volume_ma),
                    'signal': volume_signal,
                    'trend': volume_trend
                },
                # NEW: Add Bollinger Bands information
                'bollinger_bands': {
                    'upper': round(bb_upper, 2),
                    'lower': round(bb_lower, 2),
                    'width': round(bb_width, 4),
                    'signal': bb_signal,
                    'trend': bb_trend
                },
                # NEW: Add MFI information
                'money_flow_index': {
                    'value': round(mfi, 2),
                    'signal': mfi_signal,
                    'trend': mfi_trend
                },
                # NEW: Add risk metrics
                'risk_metrics': {
                    'volatility': round(volatility_pct, 2),
                    'volatility_level': volatility_level,
                    'atr': round(atr, 2),
                    'sharpe_ratio': round(sharpe_ratio, 2)
                }
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {
                'error': f"Could not calculate technical indicators: {str(e)}"
            }
    
    def get_news_sentiment(self, ticker):
        """Mock function to get news sentiment for a stock ticker."""
        try:
            # In a real implementation, this would call a news API 
            # and perform sentiment analysis on recent news articles
            
            # For demonstration, we'll generate mock data
            import random
            
            # Create mock news with varied sentiments
            sentiments = ['positive', 'neutral', 'negative']
            weights = [0.5, 0.3, 0.2]  # Higher chance of positive news
            
            # Generate random news items
            news_items = []
            for i in range(5):
                # Days ago (0-10 days)
                days_ago = random.randint(0, 10)
                date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
                
                # Random sentiment based on weights
                sentiment = random.choices(sentiments, weights=weights)[0]
                
                # Create headline based on sentiment
                if sentiment == 'positive':
                    templates = [
                        f"{ticker} Exceeds Quarterly Earnings Expectations",
                        f"Analysts Upgrade {ticker} Rating to 'Buy'",
                        f"{ticker} Announces New Product Line",
                        f"{ticker} Expands Into New Markets",
                        f"Investors Bullish on {ticker}'s Growth Prospects"
                    ]
                elif sentiment == 'neutral':
                    templates = [
                        f"{ticker} Reports Earnings In Line With Expectations",
                        f"{ticker} Maintains Market Position Despite Challenges",
                        f"Analysts Hold {ticker} Rating Steady",
                        f"{ticker} Announces Leadership Changes",
                        f"Industry Report Shows {ticker} Holding Steady"
                    ]
                else:  # negative
                    templates = [
                        f"{ticker} Misses Quarterly Earnings Expectations",
                        f"Analysts Downgrade {ticker} Rating",
                        f"{ticker} Faces Regulatory Challenges",
                        f"Market Share Decreases for {ticker}",
                        f"Investors Cautious About {ticker}'s Future"
                    ]
                
                headline = random.choice(templates)
                
                news_items.append({
                    'date': date,
                    'headline': headline,
                    'sentiment': sentiment
                })
            
            # Sort by date (most recent first)
            news_items.sort(key=lambda x: x['date'], reverse=True)
            
            # Calculate overall sentiment score (-1 to 1)
            sentiment_values = {'positive': 1, 'neutral': 0, 'negative': -1}
            sentiment_scores = [sentiment_values[item['sentiment']] for item in news_items]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            return {
                'news_items': news_items,
                'sentiment_score': round(avg_sentiment, 2),
                'sentiment_label': 'Positive' if avg_sentiment > 0.3 else 'Negative' if avg_sentiment < -0.3 else 'Neutral'
            }
        except Exception as e:
            logger.error(f"Error getting news sentiment: {str(e)}")
            return {
                'error': f"Could not get news sentiment: {str(e)}",
                'news_items': [],
                'sentiment_score': 0,
                'sentiment_label': 'Neutral'
            }
    
    def predict_stock(self, ticker, days=30):
        try:
            # Prepare and train on the data
            stock_data = self.prepare_data(ticker)
            self.train(stock_data)
            
            # Make prediction
            result = self.predict(stock_data)
            
            # Add additional information using standard Python types
            result['ticker'] = str(ticker)
            
            # Ensure proper conversion to Python types
            result['current_price'] = float(round(stock_data['Close'].iloc[-1], 2))
            result['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Store historical data for charting (last N days)
            historical_data = []
            for i in range(min(days, len(stock_data))):
                idx = len(stock_data) - 1 - i
                if idx >= 0:
                    historical_data.append({
                        'date': stock_data.index[idx].strftime('%Y-%m-%d'),
                        'price': float(round(stock_data['Close'].iloc[idx], 2))
                    })
            
            # Reverse to get chronological order
            result['historical_data'] = historical_data[::-1]
            
            # Add technical indicators
            result['technical_indicators'] = self.calculate_technical_indicators(stock_data)
            
            # NEW: Add news sentiment
            result['news_sentiment'] = self.get_news_sentiment(ticker)
            
            # Return only JSON serializable values
            return result
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error in predict_stock for {ticker}: {error_message}\n{traceback.format_exc()}")
            return {'error': error_message} 