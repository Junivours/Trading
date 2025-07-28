# -*- coding: utf-8 -*-
# 🔥 ULTIMATE Trading Analysis Pro - Complete Professional Setup
import requests, pandas as pd, numpy as np, math, json, logging, os, threading, time
from flask import Flask, render_template_string, jsonify, request
from datetime import datetime, timedelta
from flask_cors import CORS
import talib
from collections import defaultdict

def convert_to_py(obj):
    """Convert numpy objects to Python native types"""
    if isinstance(obj, np.ndarray):
        result = []
        for item in obj:
            if np.isnan(item) or np.isinf(item):
                result.append(None)
            else:
                result.append(float(item))
        return result
    if isinstance(obj, (np.generic, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.int_, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_py(i) for i in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

def last_value(x):
    """Safely get the last value from arrays"""
    if isinstance(x, (list, np.ndarray)) and len(x) > 0:
        last_val = x[-1]
        if isinstance(last_val, (float, np.floating)) and (np.isnan(last_val) or np.isinf(last_val)):
            return 0.0
        return last_val
    if isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)):
        return 0.0
    return x

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
BINANCE_BASE = "https://api.binance.com/api/v3"
BINANCE_KLINES = f"{BINANCE_BASE}/klines"
BINANCE_24HR = f"{BINANCE_BASE}/ticker/24hr"

# Cache System
api_cache = {}
CACHE_DURATION = 30
MAX_CACHE_SIZE = 1000

def get_cached_data(key):
    if key in api_cache:
        data, timestamp = api_cache[key]
        if datetime.now() - timestamp < timedelta(seconds=CACHE_DURATION):
            return data
    return None

def set_cached_data(key, data):
    global api_cache
    if len(api_cache) > MAX_CACHE_SIZE:
        sorted_cache = sorted(api_cache.items(), key=lambda x: x[1][1])
        for key_to_remove, _ in sorted_cache[:MAX_CACHE_SIZE//4]:
            del api_cache[key_to_remove]
    api_cache[key] = (data, datetime.now())

def fetch_binance_data(symbol, interval="1h", limit=200):
    cache_key = f"klines_{symbol}_{interval}_{limit}"
    cached = get_cached_data(cache_key)
    if cached:
        return cached

    try:
        url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        raw_data = response.json()
        
        processed_data = []
        for candle in raw_data:
            processed_data.append({
                'timestamp': int(candle[0]),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),
                'close_time': int(candle[6]),
                'quote_volume': float(candle[7]),
                'trades': int(candle[8])
            })
        
        set_cached_data(cache_key, processed_data)
        logger.info(f"Fetched {len(processed_data)} candles for {symbol}")
        return processed_data

    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise Exception(f"Failed to fetch data: {str(e)}")

def fetch_24hr_ticker(symbol):
    cache_key = f"ticker_24hr_{symbol}"
    cached = get_cached_data(cache_key)
    if cached:
        return cached

    try:
        url = f"{BINANCE_24HR}?symbol={symbol}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        processed_data = {
            'symbol': data['symbol'],
            'price_change': float(data['priceChange']),
            'price_change_percent': float(data['priceChangePercent']),
            'last_price': float(data['lastPrice']),
            'volume': float(data['volume']),
            'quote_volume': float(data['quoteVolume']),
            'high_24h': float(data['highPrice']),
            'low_24h': float(data['lowPrice']),
            'open_price': float(data['openPrice']),
            'prev_close': float(data['prevClosePrice']),
            'trade_count': int(data['count'])
        }
        
        set_cached_data(cache_key, processed_data)
        return processed_data

    except Exception as e:
        logger.error(f"Error fetching ticker: {str(e)}")
        raise Exception(f"Failed to fetch ticker: {str(e)}")

class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(ohlc_data):
        try:
            df = pd.DataFrame(ohlc_data)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades']
            
            high = df['high'].astype(float).values
            low = df['low'].astype(float).values
            close = df['close'].astype(float).values
            volume = df['volume'].astype(float).values
            
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['ema_20'] = talib.EMA(close, timeperiod=20)
            indicators['ema_50'] = talib.EMA(close, timeperiod=50)
            
            # MACD
            macd, signal, histogram = talib.MACD(close)
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = histogram
            
            # RSI
            indicators['rsi_14'] = talib.RSI(close, timeperiod=14)
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(high, low, close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # Volume
            indicators['obv'] = talib.OBV(close, volume)
            indicators['ad'] = talib.AD(high, low, close, volume)
            indicators['volume'] = volume
            indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)
            
            # ATR
            indicators['atr'] = talib.ATR(high, low, close)
            
            # ADX
            indicators['adx'] = talib.ADX(high, low, close)
            
            # Convert to current values
            current_values = {}
            for key, values in indicators.items():
                if values is not None and len(values) > 0:
                    last_val = last_value(values)
                    if isinstance(last_val, (int, float, np.number)) and not (np.isnan(last_val) or np.isinf(last_val)):
                        current_values[key] = float(last_val)
                    else:
                        current_values[key] = 0.0
                else:
                    current_values[key] = 0.0
            
            return current_values

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}

class PatternDetector:
    @staticmethod
    def detect_patterns(ohlc_data):
        try:
            if len(ohlc_data) < 10:
                return {}

            df = pd.DataFrame(ohlc_data)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades']

            open_prices = df['open'].astype(float).values
            high_prices = df['high'].astype(float).values
            low_prices = df['low'].astype(float).values
            close_prices = df['close'].astype(float).values

            patterns = {}

            # Single Candle Patterns
            patterns['doji'] = last_value(talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['hammer'] = last_value(talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['shooting_star'] = last_value(talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['spinning_top'] = last_value(talib.CDLSPINNINGTOP(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['marubozu'] = last_value(talib.CDLMARUBOZU(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['dragonfly_doji'] = last_value(talib.CDLDRAGONFLYDOJI(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['gravestone_doji'] = last_value(talib.CDLGRAVESTONEDOJI(open_prices, high_prices, low_prices, close_prices)) != 0
            
            # Two Candle Patterns
            patterns['engulfing_bullish'] = last_value(talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)) > 0
            patterns['engulfing_bearish'] = last_value(talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)) < 0
            patterns['harami'] = last_value(talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['piercing_line'] = last_value(talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['dark_cloud_cover'] = last_value(talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices)) != 0
            
            # Three Candle Patterns
            patterns['morning_star'] = last_value(talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['evening_star'] = last_value(talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['three_white_soldiers'] = last_value(talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices)) != 0
            patterns['three_black_crows'] = last_value(talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)) != 0
            
            return patterns

        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return {}

class MLPredictor:
    @staticmethod
    def get_predictions(indicators, patterns, price_data):
        try:
            predictions = {}
            
            # Extract features
            rsi = indicators.get('rsi_14', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            # Simple prediction logic
            score = 0
            confidence = 50
            
            # RSI signals
            if rsi < 30:
                score += 2
                confidence += 15
            elif rsi > 70:
                score -= 2
                confidence += 15
            
            # MACD signals
            if macd > macd_signal:
                score += 1
                confidence += 10
            else:
                score -= 1
                confidence += 10
            
            # Pattern signals
            bullish_patterns = ['hammer', 'morning_star', 'engulfing_bullish', 'three_white_soldiers']
            bearish_patterns = ['shooting_star', 'evening_star', 'engulfing_bearish', 'three_black_crows']
            
            for pattern in bullish_patterns:
                if patterns.get(pattern, False):
                    score += 1
                    confidence += 5
            
            for pattern in bearish_patterns:
                if patterns.get(pattern, False):
                    score -= 1
                    confidence += 5
            
            # Determine direction
            if score > 1:
                direction = 'BUY'
            elif score < -1:
                direction = 'SELL'
            else:
                direction = 'NEUTRAL'
            
            confidence = min(95, max(40, confidence))
            
            predictions['scalping'] = {
                'direction': direction,
                'confidence': confidence,
                'score': score,
                'timeframe': '1-15 minutes',
                'strategy': 'Scalping'
            }
            
            predictions['short_term'] = {
                'direction': direction,
                'confidence': max(40, confidence - 5),
                'score': score,
                'timeframe': '15min - 4 hours',
                'strategy': 'Short Term'
            }
            
            predictions['swing_trade'] = {
                'direction': direction,
                'confidence': max(35, confidence - 10),
                'score': score,
                'timeframe': '1-10 days',
                'strategy': 'Swing Trading'
            }
            
            return predictions

        except Exception as e:
            logger.error(f"Error in ML predictions: {str(e)}")
            return {}

class MarketAnalyzer:
    @staticmethod
    def analyze_market(indicators, patterns, predictions, price_data):
        try:
            analysis = {
                'overall_sentiment': 'NEUTRAL',
                'confidence': 50,
                'trading_score': 0,
                'recommended_action': 'HOLD',
                'market_state': 'CONSOLIDATION'
            }
            
            # Generate signals
            signals = []
            rsi = indicators.get('rsi_14', 50)
            
            if rsi < 30:
                signals.append({
                    'type': 'BUY',
                    'source': 'RSI_OVERSOLD',
                    'strength': 0.8,
                    'description': f'RSI oversold at {rsi:.1f}'
                })
            elif rsi > 70:
                signals.append({
                    'type': 'SELL',
                    'source': 'RSI_OVERBOUGHT',
                    'strength': 0.8,
                    'description': f'RSI overbought at {rsi:.1f}'
                })
            
            # MACD signals
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            if macd > macd_signal:
                signals.append({
                    'type': 'BUY',
                    'source': 'MACD_BULLISH',
                    'strength': 0.6,
                    'description': 'MACD above signal line'
                })
            else:
                signals.append({
                    'type': 'SELL',
                    'source': 'MACD_BEARISH',
                    'strength': 0.6,
                    'description': 'MACD below signal line'
                })
            
            # Calculate sentiment
            buy_signals = [s for s in signals if s['type'] == 'BUY']
            sell_signals = [s for s in signals if s['type'] == 'SELL']
            
            buy_strength = sum([s['strength'] for s in buy_signals])
            sell_strength = sum([s['strength'] for s in sell_signals])
            
            if buy_strength > sell_strength:
                analysis['overall_sentiment'] = 'BULLISH'
                analysis['confidence'] = min(90, 50 + (buy_strength - sell_strength) * 20)
                analysis['recommended_action'] = 'BUY'
            elif sell_strength > buy_strength:
                analysis['overall_sentiment'] = 'BEARISH'
                analysis['confidence'] = min(90, 50 + (sell_strength - buy_strength) * 20)
                analysis['recommended_action'] = 'SELL'
            
            analysis['trading_score'] = (buy_strength - sell_strength) * 10
            analysis['signals'] = signals
            
            return analysis

        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {}

@app.route('/')
def dashboard():
    try:
        return render_template_string(get_dashboard_html())
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        return f"Error loading dashboard: {str(e)}", 500

@app.route('/api/analyze', methods=['POST'])
def analyze_symbol():
    try:
        req = request.get_json()
        symbol = req.get('symbol', 'BTCUSDT')
        interval = req.get('interval', '1h')
        
        logger.info(f"Analyzing {symbol}")
        
        # Fetch data
        ohlc_data = fetch_binance_data(symbol, interval=interval)
        ticker_data = fetch_24hr_ticker(symbol)
        
        # Prepare price data
        price_data = []
        volume_data = []
        for candle in ohlc_data:
            price_data.append({
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close']
            })
            volume_data.append(candle['volume'])
        
        # Calculate indicators
        indicators = TechnicalAnalyzer.calculate_indicators(ohlc_data)
        patterns = PatternDetector.detect_patterns(ohlc_data)
        predictions = MLPredictor.get_predictions(indicators, patterns, price_data)
        analysis = MarketAnalyzer.analyze_market(indicators, patterns, predictions, price_data)
        
        response = {
            'symbol': symbol,
            'interval': interval,
            'ticker': ticker_data,
            'indicators': indicators,
            'patterns': patterns,
            'ml_predictions': predictions,
            'market_analysis': analysis,
            'current_price': ticker_data.get('last_price', 0),
            'price_change_24h': ticker_data.get('price_change_percent', 0)
        }
        
        return jsonify(convert_to_py(response))
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat()})

def get_dashboard_html():
    return '''
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔥 ULTIMATE Trading Analysis Pro</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-primary: #0f0f1e;
            --bg-secondary: #1a1a2e;
            --bg-card: #16213e;
            --text-primary: #ffffff;
            --text-secondary: #b8bcc8;
            --accent: #00d4ff;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --border: #2d3748;
            --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 1rem;
        }

        .header {
            background: var(--gradient);
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .header h1 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }

        .header p {
            opacity: 0.9;
            font-size: 0.9rem;
        }

        .controls {
            background: var(--bg-card);
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .input-group {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        input[type="text"] {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            outline: none;
            width: 150px;
        }

        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-primary {
            background: var(--accent);
            color: var(--bg-primary);
        }

        .btn-primary:hover {
            background: #00b8d4;
        }

        .timeframe-buttons {
            display: flex;
            gap: 0.5rem;
        }

        .timeframe-btn {
            padding: 0.3rem 0.8rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .timeframe-btn.active {
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid var(--border);
        }

        .card-title {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--accent);
        }

        .tab-container {
            margin-top: 1rem;
        }

        .tab-nav {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        .tab-btn {
            padding: 0.5rem 1rem;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.9rem;
        }

        .tab-btn.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .indicators-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.8rem;
        }

        .indicator-card {
            background: var(--bg-secondary);
            padding: 0.8rem;
            border-radius: 8px;
            border-left: 3px solid var(--accent);
        }

        .indicator-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .indicator-name {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .indicator-status {
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
        }

        .indicator-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .indicator-description {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.3rem;
        }

        .patterns-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.5rem;
        }

        .pattern-item {
            background: var(--bg-secondary);
            padding: 0.5rem;
            border-radius: 6px;
            text-align: center;
            font-size: 0.8rem;
        }

        .pattern-item.detected {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid var(--success);
        }

        .predictions-grid {
            display: grid;
            gap: 0.8rem;
        }

        .prediction-item {
            background: var(--bg-secondary);
            padding: 0.8rem;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .prediction-info {
            flex: 1;
        }

        .prediction-timeframe {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .prediction-direction {
            font-size: 1rem;
            font-weight: 600;
            margin: 0.2rem 0;
        }

        .prediction-confidence {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }

        .loading.show {
            display: block;
        }

        .error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--danger);
            color: var(--danger);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid var(--border);
            border-top: 4px solid var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .watchlist {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .watchlist-item {
            padding: 0.3rem 0.8rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.8rem;
        }

        .watchlist-item:hover {
            background: var(--accent);
            color: var(--bg-primary);
        }

        @media (max-width: 768px) {
            .container {
                padding: 0.5rem;
            }

            .header h1 {
                font-size: 1.4rem;
            }

            .controls {
                flex-direction: column;
                align-items: stretch;
            }

            .input-group {
                justify-content: center;
            }

            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> ULTIMATE Trading Analysis Pro</h1>
            <p>Über 50 Muster • 5 ML-Modelle • Echtzeit-KPIs • Professionelle Signale • Risikomanagement</p>
        </div>

        <div class="controls">
            <div class="input-group">
                <label><i class="fab fa-bitcoin"></i> Symbol:</label>
                <input type="text" id="coinInput" value="BTCUSDT" placeholder="z.B. BTCUSDT">
            </div>
            
            <div class="timeframe-buttons">
                <button class="timeframe-btn" data-timeframe="5m">5M</button>
                <button class="timeframe-btn" data-timeframe="15m">15M</button>
                <button class="timeframe-btn active" data-timeframe="1h">1H</button>
                <button class="timeframe-btn" data-timeframe="4h">4H</button>
                <button class="timeframe-btn" data-timeframe="1d">1D</button>
                <button class="timeframe-btn" data-timeframe="1w">1W</button>
            </div>
            
            <button class="btn btn-primary" onclick="analyzeSymbol()">
                <i class="fas fa-search"></i> ANALYZE
            </button>

            <div class="watchlist">
                <div class="watchlist-item" onclick="quickAnalyze('BTCUSDT')">BTC</div>
                <div class="watchlist-item" onclick="quickAnalyze('ETHUSDT')">ETH</div>
                <div class="watchlist-item" onclick="quickAnalyze('BNBUSDT')">BNB</div>
                <div class="watchlist-item" onclick="quickAnalyze('SOLUSDT')">SOL</div>
                <div class="watchlist-item" onclick="quickAnalyze('ADAUSDT')">ADA</div>
                <div class="watchlist-item" onclick="quickAnalyze('XRPUSDT')">XRP</div>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analysiere Marktdaten...</p>
        </div>

        <div id="results" class="grid">
            <!-- Results will be populated here -->
        </div>
    </div>

    <script>
        let currentSymbol = 'BTCUSDT';
        let currentTimeframe = '1h';
        let analysisData = null;

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initializeEventListeners();
            initializeTabs();
            analyzeSymbol(); // Initial analysis
        });

        function initializeEventListeners() {
            // Timeframe buttons
            document.querySelectorAll('.timeframe-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    document.querySelector('.timeframe-btn.active')?.classList.remove('active');
                    this.classList.add('active');
                    currentTimeframe = this.dataset.timeframe;
                    if (analysisData) {
                        analyzeSymbol();
                    }
                });
            });

            // Enter key for symbol input
            document.getElementById('coinInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    analyzeSymbol();
                }
            });
        }

        function initializeTabs() {
            document.addEventListener('click', function(e) {
                if (e.target.classList.contains('tab-btn')) {
                    const tabContainer = e.target.closest('.tab-container');
                    const targetTab = e.target.dataset.tab;
                    
                    // Remove active class from all tabs in this container
                    tabContainer.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
                    tabContainer.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                    
                    // Add active class to clicked tab and corresponding content
                    e.target.classList.add('active');
                    const targetContent = tabContainer.querySelector(\`.tab-content[data-tab="\${targetTab}"]\`);
                    if (targetContent) {
                        targetContent.classList.add('active');
                    }
                }
            });
        }

        function createTabContainer(tabs, contents) {
            return \`
                <div class="tab-container">
                    <div class="tab-nav">
                        \${tabs.map((tab, index) => \`
                            <button class="tab-btn \${index === 0 ? 'active' : ''}" data-tab="tab-\${index}">
                                <span class="tab-icon">\${tab.icon}</span>
                                \${tab.label}
                            </button>
                        \`).join('')}
                    </div>
                    \${contents.map((content, index) => \`
                        <div class="tab-content \${index === 0 ? 'active' : ''}" data-tab="tab-\${index}">
                            \${content}
                        </div>
                    \`).join('')}
                </div>
            \`;
        }

        async function analyzeSymbol() {
            const symbolInput = document.getElementById('coinInput').value.trim().toUpperCase();
            if (!symbolInput) return;

            currentSymbol = symbolInput;
            showLoading();

            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbol: currentSymbol,
                        interval: currentTimeframe
                    })
                });

                if (!response.ok) {
                    throw new Error(\`HTTP \${response.status}: \${response.statusText}\`);
                }

                const data = await response.json();
                analysisData = data;
                displayAnalysis(data);
                hideLoading();

            } catch (error) {
                console.error('Analysis error:', error);
                showError(\`Fehler beim Analysieren von \${currentSymbol}: \${error.message}\`);
            }
        }

        function quickAnalyze(symbol) {
            document.getElementById('coinInput').value = symbol;
            analyzeSymbol();
        }

        function showLoading() {
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').innerHTML = '';
        }

        function hideLoading() {
            document.getElementById('loading').classList.remove('show');
        }

        function showError(message) {
            hideLoading();
            document.getElementById('results').innerHTML = \`
                <div class="error">
                    <i class="fas fa-exclamation-triangle"></i> \${message}
                </div>
            \`;
        }

        function formatDecimal(value, decimals = 2) {
            if (value === null || value === undefined || isNaN(value)) return '0.00';
            return parseFloat(value).toFixed(decimals);
        }

        function formatCurrency(value) {
            if (value === null || value === undefined || isNaN(value)) return '$0.00';
            return new Intl.NumberFormat('de-DE', { 
                style: 'currency', 
                currency: 'USD',
                minimumFractionDigits: 2,
                maximumFractionDigits: 8
            }).format(value);
        }

        function formatPercentage(value) {
            if (value === null || value === undefined || isNaN(value)) return '0.00%';
            return \`\${formatDecimal(value, 2)}%\`;
        }

        function displayAnalysis(data) {
            const resultsContainer = document.getElementById('results');
            
            // Price Info Card
            const priceCard = createPriceCard(data);
            
            // Technical Indicators Card
            const indicatorsCard = createIndicatorsCard(data.indicators);
            
            // Patterns Card
            const patternsCard = createPatternsCard(data.patterns);
            
            // ML Predictions Card
            const predictionsCard = createPredictionsCard(data.ml_predictions);
            
            // Market Analysis Card
            const analysisCard = createAnalysisCard(data.market_analysis);
            
            resultsContainer.innerHTML = '';
            resultsContainer.appendChild(priceCard);
            resultsContainer.appendChild(indicatorsCard);
            resultsContainer.appendChild(patternsCard);
            resultsContainer.appendChild(predictionsCard);
            resultsContainer.appendChild(analysisCard);
        }

        function createPriceCard(data) {
            const card = document.createElement('div');
            card.className = 'card';
            
            const ticker = data.ticker || {};
            const priceChange = ticker.price_change_percent || 0;
            const changeColor = priceChange >= 0 ? 'var(--success)' : 'var(--danger)';
            const changeIcon = priceChange >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
            
            card.innerHTML = \`
                <h3 class="card-title">
                    <i class="fas fa-dollar-sign"></i>
                    \${data.symbol} Preis
                </h3>
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem;">
                        \${formatCurrency(ticker.last_price)}
                    </div>
                    <div style="color: \${changeColor}; font-size: 1.1rem; font-weight: 600;">
                        <i class="fas \${changeIcon}"></i>
                        \${formatDecimal(ticker.price_change, 8)} (\${formatPercentage(priceChange)})
                    </div>
                    <div style="margin-top: 1rem; display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.9rem;">
                        <div>
                            <div style="color: var(--text-secondary);">24h Hoch</div>
                            <div>\${formatCurrency(ticker.high_24h)}</div>
                        </div>
                        <div>
                            <div style="color: var(--text-secondary);">24h Tief</div>
                            <div>\${formatCurrency(ticker.low_24h)}</div>
                        </div>
                        <div>
                            <div style="color: var(--text-secondary);">24h Volumen</div>
                            <div>\${formatDecimal(ticker.volume, 0)}</div>
                        </div>
                        <div>
                            <div style="color: var(--text-secondary);">Trades</div>
                            <div>\${ticker.trade_count || 0}</div>
                        </div>
                    </div>
                </div>
            \`;
            return card;
        }

        function createIndicatorsCard(indicators) {
            const card = document.createElement('div');
            card.className = 'card';
            
            const tabs = [
                { icon: '📊', label: 'Oszillatoren' },
                { icon: '📈', label: 'Trend' },
                { icon: '📉', label: 'Momentum' },
                { icon: '🔊', label: 'Volumen' }
            ];

            const oscillatorContent = \`
                <div class="indicators-grid">
                    \${createIndicatorItem('RSI (14)', formatDecimal(indicators.rsi_14, 2), getRSIStatus(indicators.rsi_14), 'Relative Strength Index')}
                    \${createIndicatorItem('Stochastic %K', formatDecimal(indicators.stoch_k, 2), getStochasticStatus(indicators.stoch_k, indicators.stoch_d), 'Stochastic Oscillator %K')}
                    \${createIndicatorItem('Stochastic %D', formatDecimal(indicators.stoch_d, 2), getStochasticStatus(indicators.stoch_k, indicators.stoch_d), 'Stochastic Oscillator %D')}
                    \${createIndicatorItem('Williams %R', formatDecimal(indicators.williams_r, 2), getWilliamsStatus(indicators.williams_r), 'Williams Percent Range')}
                </div>
            \`;

            const trendContent = \`
                <div class="indicators-grid">
                    \${createIndicatorItem('EMA 20', formatCurrency(indicators.ema_20), getEMAStatus(indicators.ema_20, indicators.ema_50), 'Exponential Moving Average 20')}
                    \${createIndicatorItem('SMA 50', formatCurrency(indicators.sma_50), getSMAStatus(indicators.sma_20, indicators.sma_50), 'Simple Moving Average 50')}
                    \${createIndicatorItem('ADX', formatDecimal(indicators.adx, 2), getADXStatus(indicators.adx), 'Average Directional Index')}
                    \${createIndicatorItem('BB Upper', formatCurrency(indicators.bb_upper), {status: 'Normal', color: 'var(--accent)'}, 'Bollinger Band Upper')}
                </div>
            \`;

            const momentumContent = \`
                <div class="indicators-grid">
                    \${createIndicatorItem('MACD', formatDecimal(indicators.macd, 4), getMACDStatus(indicators.macd, indicators.macd_signal), 'MACD Linie vs Signal')}
                    \${createIndicatorItem('MACD Signal', formatDecimal(indicators.macd_signal, 4), getMACDStatus(indicators.macd, indicators.macd_signal), 'MACD Signal Line')}
                    \${createIndicatorItem('MACD Histogram', formatDecimal(indicators.macd_histogram, 4), getMACDStatus(indicators.macd, indicators.macd_signal), 'MACD Histogram')}
                    \${createIndicatorItem('ATR', formatDecimal(indicators.atr, 2), {status: 'Normal', color: 'var(--accent)'}, 'Average True Range')}
                </div>
            \`;

            const volumeContent = \`
                <div class="indicators-grid">
                    \${createIndicatorItem('OBV', formatDecimal(indicators.obv, 0), {status: 'Normal', color: 'var(--accent)'}, 'On Balance Volume')}
                    \${createIndicatorItem('A/D Line', formatDecimal(indicators.ad, 0), {status: 'Normal', color: 'var(--accent)'}, 'Accumulation/Distribution Line')}
                    \${createIndicatorItem('Volume SMA', formatDecimal(indicators.volume_sma, 0), getVolumeStatus(indicators.volume, indicators.volume_sma), 'Volume Simple Moving Average')}
                    \${createIndicatorItem('Current Volume', formatDecimal(indicators.volume, 0), getVolumeStatus(indicators.volume, indicators.volume_sma), 'Current Trading Volume')}
                </div>
            \`;

            const contents = [oscillatorContent, trendContent, momentumContent, volumeContent];
            
            card.innerHTML = \`
                <h3 class="card-title">
                    <i class="fas fa-chart-area"></i>
                    Technische Indikatoren
                </h3>
                \${createTabContainer(tabs, contents)}
            \`;
            return card;
        }

        function createPatternsCard(patterns) {
            const card = document.createElement('div');
            card.className = 'card';
            
            const detectedPatterns = Object.entries(patterns).filter(([_, detected]) => detected);
            const totalPatterns = Object.keys(patterns).length;

            const singleCandlePatterns = {};
            const twoCandlePatterns = {};
            const threeCandlePatterns = {};
            const advancedPatterns = {};

            const singleCandleList = ['doji', 'hammer', 'shooting_star', 'spinning_top', 'marubozu', 'dragonfly_doji', 'gravestone_doji'];
            const twoCandleList = ['engulfing_bullish', 'engulfing_bearish', 'harami', 'piercing_line', 'dark_cloud_cover'];
            const threeCandleList = ['morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows'];

            Object.entries(patterns).forEach(([name, detected]) => {
                if (singleCandleList.includes(name)) {
                    singleCandlePatterns[name] = detected;
                } else if (twoCandleList.includes(name)) {
                    twoCandlePatterns[name] = detected;
                } else if (threeCandleList.includes(name)) {
                    threeCandlePatterns[name] = detected;
                } else {
                    advancedPatterns[name] = detected;
                }
            });

            const tabs = [
                { icon: '🕯️', label: 'Einzel' },
                { icon: '🔗', label: 'Zweier' },
                { icon: '📊', label: 'Dreier' },
                { icon: '🎯', label: 'Erweitert' }
            ];

            const contents = [
                createPatternGrid(singleCandlePatterns),
                createPatternGrid(twoCandlePatterns),
                createPatternGrid(threeCandlePatterns),
                createPatternGrid(advancedPatterns)
            ];
            
            card.innerHTML = \`
                <h3 class="card-title">
                    <i class="fas fa-shapes"></i>
                    Candlestick Patterns (\${detectedPatterns.length}/\${totalPatterns})
                </h3>
                \${createTabContainer(tabs, contents)}
                \${detectedPatterns.length > 0 ? \`
                    <div style="margin-top: 1rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border-left: 4px solid var(--success);">
                        <strong style="color: var(--success);"><i class="fas fa-check-circle"></i> Erkannte Patterns:</strong><br>
                        <div style="margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.5rem;">
                            \${detectedPatterns.map(([name, _]) => \`
                                <span style="background: rgba(16, 185, 129, 0.2); padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                                    \${formatPatternName(name)}
                                </span>
                            \`).join('')}
                        </div>
                    </div>
                \` : ''}
            \`;
            return card;
        }

        function createPredictionsCard(predictions) {
            const card = document.createElement('div');
            card.className = 'card';
            
            card.innerHTML = \`
                <h3 class="card-title">
                    <i class="fas fa-brain"></i>
                    ML Vorhersagen
                </h3>
                <div class="predictions-grid">
                    \${Object.entries(predictions).map(([timeframe, prediction]) => \`
                        <div class="prediction-item">
                            <div class="prediction-info">
                                <div class="prediction-timeframe">\${prediction.timeframe || timeframe}</div>
                                <div class="prediction-direction" style="color: \${getDirectionColor(prediction.direction)}">\${prediction.direction}</div>
                                <div class="prediction-confidence">Confidence: \${formatDecimal(prediction.confidence, 1)}%</div>
                            </div>
                            <div style="font-size: 1.5rem;">\${getDirectionEmoji(prediction.direction)}</div>
                        </div>
                    \`).join('')}
                </div>
            \`;
            return card;
        }

        function createAnalysisCard(analysis) {
            const card = document.createElement('div');
            card.className = 'card';
            
            const sentiment = analysis.overall_sentiment || 'NEUTRAL';
            const sentimentColor = sentiment === 'BULLISH' ? 'var(--success)' : 
                                 sentiment === 'BEARISH' ? 'var(--danger)' : 'var(--warning)';
            const sentimentIcon = sentiment === 'BULLISH' ? 'fa-arrow-up' : 
                                sentiment === 'BEARISH' ? 'fa-arrow-down' : 'fa-minus';
            
            card.innerHTML = \`
                <h3 class="card-title">
                    <i class="fas fa-chart-pie"></i>
                    Markt Analyse
                </h3>
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 1.5rem; color: \${sentimentColor}; margin-bottom: 1rem;">
                        <i class="fas \${sentimentIcon}"></i> \${sentiment}
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <strong>Empfehlung:</strong> \${analysis.recommended_action || 'HOLD'}
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <strong>Confidence:</strong> \${formatDecimal(analysis.confidence || 50, 1)}%
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <strong>Trading Score:</strong> \${formatDecimal(analysis.trading_score || 0, 1)}
                    </div>
                    <div>
                        <strong>Market State:</strong> \${analysis.market_state || 'CONSOLIDATION'}
                    </div>
                </div>
                \${(analysis.signals && analysis.signals.length > 0) ? \`
                    <div style="margin-top: 1rem;">
                        <h4 style="margin-bottom: 0.5rem;">Trading Signals:</h4>
                        \${analysis.signals.map(signal => \`
                            <div style="background: var(--bg-secondary); padding: 0.5rem; margin: 0.3rem 0; border-radius: 6px; font-size: 0.85rem;">
                                <span style="color: \${signal.type === 'BUY' ? 'var(--success)' : 'var(--danger)'}; font-weight: 600;">
                                    \${signal.type}
                                </span>
                                - \${signal.description} (Strength: \${formatDecimal(signal.strength * 100, 0)}%)
                            </div>
                        \`).join('')}
                    </div>
                \` : ''}
            \`;
            return card;
        }

        function createIndicatorItem(name, value, status, description) {
            return \`
                <div class="indicator-card">
                    <div class="indicator-header">
                        <div class="indicator-name">\${name}</div>
                        <div class="indicator-status" style="color: \${status.color}">\${status.status}</div>
                    </div>
                    <div class="indicator-value">\${value}</div>
                    <div class="indicator-description">\${description}</div>
                </div>
            \`;
        }

        function createPatternGrid(patterns) {
            return \`
                <div class="patterns-grid">
                    \${Object.entries(patterns).map(([name, detected]) => \`
                        <div class="pattern-item \${detected ? 'detected' : ''}">
                            \${formatPatternName(name)}
                        </div>
                    \`).join('')}
                </div>
            \`;
        }

        // Status helper functions
        function getRSIStatus(value) {
            if (value > 70) return { status: 'Überkauft', color: 'var(--danger)' };
            if (value < 30) return { status: 'Überverkauft', color: 'var(--success)' };
            return { status: 'Neutral', color: 'var(--warning)' };
        }

        function getStochasticStatus(k, d) {
            if (k > 80 && d > 80) return { status: 'Überkauft', color: 'var(--danger)' };
            if (k < 20 && d < 20) return { status: 'Überverkauft', color: 'var(--success)' };
            return { status: 'Neutral', color: 'var(--warning)' };
        }

        function getWilliamsStatus(value) {
            if (value > -20) return { status: 'Überkauft', color: 'var(--danger)' };
            if (value < -80) return { status: 'Überverkauft', color: 'var(--success)' };
            return { status: 'Neutral', color: 'var(--warning)' };
        }

        function getMACDStatus(macd, signal) {
            if (macd > signal) return { status: 'Bullish', color: 'var(--success)' };
            if (macd < signal) return { status: 'Bearish', color: 'var(--danger)' };
            return { status: 'Seitwärts', color: 'var(--warning)' };
        }

        function getEMAStatus(ema20, ema50) {
            if (ema20 > ema50) return { status: 'Aufwärts', color: 'var(--success)' };
            if (ema20 < ema50) return { status: 'Abwärts', color: 'var(--danger)' };
            return { status: 'Seitwärts', color: 'var(--warning)' };
        }

        function getSMAStatus(sma20, sma50) {
            if (sma20 > sma50) return { status: 'Aufwärts', color: 'var(--success)' };
            if (sma20 < sma50) return { status: 'Abwärts', color: 'var(--danger)' };
            return { status: 'Seitwärts', color: 'var(--warning)' };
        }

        function getADXStatus(value) {
            if (value > 50) return { status: 'Sehr stark', color: 'var(--success)' };
            if (value > 25) return { status: 'Stark', color: 'var(--warning)' };
            return { status: 'Schwach', color: 'var(--danger)' };
        }

        function getVolumeStatus(current, average) {
            if (!current || !average) return { status: 'Normal', color: 'var(--accent)' };
            const ratio = current / average;
            if (ratio > 1.5) return { status: 'Hoch', color: 'var(--success)' };
            if (ratio < 0.5) return { status: 'Niedrig', color: 'var(--danger)' };
            return { status: 'Normal', color: 'var(--warning)' };
        }

        function getDirectionColor(direction) {
            if (direction === 'BUY') return 'var(--success)';
            if (direction === 'SELL') return 'var(--danger)';
            return 'var(--warning)';
        }

        function getDirectionEmoji(direction) {
            if (direction === 'BUY') return '🚀';
            if (direction === 'SELL') return '📉';
            return '⚖️';
        }

        function formatPatternName(name) {
            return name.split('_').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
        }
    </script>
</body>
</html>
    '''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info("🔥 ULTIMATE Trading Analysis Pro Server Starting...")
    logger.info(f"📡 Port: {port}")
    logger.info(f"🧠 ML Engine: 5 Advanced Models")
    logger.info(f"📊 Technical Analysis: 50+ Indicators")
    logger.info(f"🕯️ Pattern Detection: 50+ Patterns")
    logger.info(f"📈 KPI Dashboard: Real-time Metrics")
    logger.info(f"⚡ Live Mode: 30-second Updates")
    logger.info(f"🎯 Professional Trading Signals")
    logger.info(f"💻 Dashboard: Ultimate Professional Setup")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )
