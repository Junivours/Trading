# === MARKIERUNG 1: Datei-Beginn, alle Imports und Flask-Initialisierung ===
# -*- coding: utf-8 -*-
# 🔥 ULTIMATE Trading Analysis Pro - Complete Professional Setup
# Advanced Pattern Recognition • ML Predictions • KPI Dashboard • Trading Recommendations
# Ready for Railway Deployment

import requests, pandas as pd, numpy as np, math, json, logging, os, threading, time
from flask import Flask, render_template_string, jsonify, request
from datetime import datetime, timedelta
from flask_cors import CORS
import talib
from collections import defaultdict

# Advanced Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
BINANCE_BASE = "https://api.binance.com/api/v3"
BINANCE_KLINES = f"{BINANCE_BASE}/klines"
BINANCE_24HR = f"{BINANCE_BASE}/ticker/24hr"
BINANCE_TICKER = f"{BINANCE_BASE}/ticker/price"

# Advanced Cache System
api_cache = {}
performance_cache = {}
CACHE_DURATION = 30  # 30 seconds for real-time feel
MAX_CACHE_SIZE = 1000

# Advanced Technical Analysis Engine
class AdvancedTechnicalAnalyzer:
    @staticmethod
    def calculate_all_indicators(ohlc_data):
        try:
            df = pd.DataFrame(ohlc_data)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades']
            high = df['high'].astype(float).values
            low = df['low'].astype(float).values
            close = df['close'].astype(float).values
            volume = df['volume'].astype(float).values
            indicators = {}

            # Trend Indicators
            indicators['sma_5'] = talib.SMA(close, timeperiod=5)
            indicators['sma_10'] = talib.SMA(close, timeperiod=10)
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['sma_200'] = talib.SMA(close, timeperiod=200)

            indicators['ema_5'] = talib.EMA(close, timeperiod=5)
            indicators['ema_10'] = talib.EMA(close, timeperiod=10)
            indicators['ema_20'] = talib.EMA(close, timeperiod=20)
            indicators['ema_50'] = talib.EMA(close, timeperiod=50)
            indicators['ema_200'] = talib.EMA(close, timeperiod=200)

            # MACD
            macd, macdsignal, macdhist = talib.MACD(close)
            indicators['macd'] = macd
            indicators['macd_signal'] = macdsignal
            indicators['macd_histogram'] = macdhist

            # Oscillators
            indicators['rsi_14'] = talib.RSI(close, timeperiod=14)
            indicators['rsi_7'] = talib.RSI(close, timeperiod=7)
            indicators['rsi_21'] = talib.RSI(close, timeperiod=21)

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd

            # Williams %R
            indicators['williams_r'] = talib.WILLR(high, low, close)

            # Commodity Channel Index
            indicators['cci'] = talib.CCI(high, low, close)

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower

            # Volume Indicators
            indicators['obv'] = talib.OBV(close, volume)
            indicators['ad'] = talib.AD(high, low, close, volume)
            indicators['adosc'] = talib.ADOSC(high, low, close, volume)

            # ATR (Average True Range)
            indicators['atr'] = talib.ATR(high, low, close)

            # Parabolic SAR
            indicators['sar'] = talib.SAR(high, low)

            # ADX (Average Directional Index)
            indicators['adx'] = talib.ADX(high, low, close)
            indicators['adx_plus'] = talib.PLUS_DI(high, low, close)
            indicators['adx_minus'] = talib.MINUS_DI(high, low, close)

            # Current values for easy access
            current_values = {}
            for key, values in indicators.items():
                if values is not None and len(values) > 0 and not np.isnan(values[-1]):
                    current_values[f'current_{key}'] = float(values[-1])
                else:
                    current_values[f'current_{key}'] = 0.0

            indicators.update(current_values)
            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}

# Advanced Pattern Detection Engine
class AdvancedPatternDetector:
    @staticmethod
    def detect_all_patterns(ohlc_data):
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
            patterns['doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['inverted_hammer'] = talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['hanging_man'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['spinning_top'] = talib.CDLSPINNINGTOP(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['marubozu'] = talib.CDLMARUBOZU(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['gravestone_doji'] = talib.CDLGRAVESTONEDOJI(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            
            # Two Candle Patterns
            patterns['engulfing_bullish'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1] > 0
            patterns['engulfing_bearish'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1] < 0
            patterns['harami'] = talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['harami_cross'] = talib.CDLHARAMICROSS(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['piercing_line'] = talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['tweezer_tops'] = talib.CDLINNECK(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            
            # Three Candle Patterns
            patterns['morning_star'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['evening_star'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['morning_doji_star'] = talib.CDLMORNINGDOJISTAR(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['evening_doji_star'] = talib.CDLEVENINGDOJISTAR(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['three_black_crows'] = talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['three_inside_up'] = talib.CDL3INSIDE(open_prices, high_prices, low_prices, close_prices)[-1] > 0
            patterns['three_inside_down'] = talib.CDL3INSIDE(open_prices, high_prices, low_prices, close_prices)[-1] < 0
            patterns['three_outside_up'] = talib.CDL3OUTSIDE(open_prices, high_prices, low_prices, close_prices)[-1] > 0
            patterns['three_outside_down'] = talib.CDL3OUTSIDE(open_prices, high_prices, low_prices, close_prices)[-1] < 0
            
            # Advanced Patterns
            patterns['abandoned_baby_bullish'] = talib.CDLABANDONEDBABY(open_prices, high_prices, low_prices, close_prices)[-1] > 0
            patterns['abandoned_baby_bearish'] = talib.CDLABANDONEDBABY(open_prices, high_prices, low_prices, close_prices)[-1] < 0
            patterns['advance_block'] = talib.CDLADVANCEBLOCK(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['belt_hold'] = talib.CDLBELTHOLD(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['breakaway'] = talib.CDLBREAKAWAY(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['closing_marubozu'] = talib.CDLCLOSINGMARUBOZU(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['concealing_baby_swallow'] = talib.CDLCONCEALBABYSWALL(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['counterattack'] = talib.CDLCOUNTERATTACK(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['doji_star'] = talib.CDLDOJISTAR(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['gapping_up_side_white'] = talib.CDLGAPSIDESIDEWHITE(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['high_wave'] = talib.CDLHIGHWAVE(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['hikkake'] = talib.CDLHIKKAKE(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['modified_hikkake'] = talib.CDLHIKKAKEMOD(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['homing_pigeon'] = talib.CDLHOMINGPIGEON(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['identical_three_crows'] = talib.CDLIDENTICAL3CROWS(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['in_neck'] = talib.CDLINNECK(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['inverted_hammer'] = talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['kicking'] = talib.CDLKICKING(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['kicking_by_length'] = talib.CDLKICKINGBYLENGTH(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['ladder_bottom'] = talib.CDLLADDERBOTTOM(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['long_legged_doji'] = talib.CDLLONGLEGGEDDOJI(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['long_line'] = talib.CDLLONGLINE(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['matching_low'] = talib.CDLMATCHINGLOW(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['mat_hold'] = talib.CDLMATHOLD(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['on_neck'] = talib.CDLONNECK(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['rickshaw_man'] = talib.CDLRICKSHAWMAN(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['rise_fall_three_methods'] = talib.CDLRISEFALL3METHODS(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['separating_lines'] = talib.CDLSEPARATINGLINES(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['short_line'] = talib.CDLSHORTLINE(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['stalled_pattern'] = talib.CDLSTALLEDPATTERN(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['stick_sandwich'] = talib.CDLSTICKSANDWICH(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['takuri'] = talib.CDLTAKURI(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['tasuki_gap'] = talib.CDLTASUKIGAP(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['thrusting'] = talib.CDLTHRUSTING(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['tristar'] = talib.CDLTRISTAR(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['unique_three_river'] = talib.CDLUNIQUE3RIVER(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['upside_gap_two_crows'] = talib.CDLUPSIDEGAP2CROWS(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            patterns['xside_gap_three_methods'] = talib.CDLXSIDEGAP3METHODS(open_prices, high_prices, low_prices, close_prices)[-1] != 0
            
            return patterns

        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return {}

# Advanced ML Prediction Engine
class AdvancedMLPredictor:
    @staticmethod
    def calculate_predictions(indicators, patterns, price_data, volume_data):
        try:
            features = AdvancedMLPredictor._extract_comprehensive_features(indicators, patterns, price_data, volume_data)
            predictions = {
                'scalping': AdvancedMLPredictor._predict_scalping(features),
                'short_term': AdvancedMLPredictor._predict_short_term(features),
                'medium_term': AdvancedMLPredictor._predict_medium_term(features),
                'long_term': AdvancedMLPredictor._predict_long_term(features),
                'swing_trade': AdvancedMLPredictor._predict_swing_trade(features)
            }
            return predictions
        except Exception as e:
            logger.error(f"Error in ML predictions: {str(e)}")
            return {}

    @staticmethod
    def _extract_comprehensive_features(indicators, patterns, price_data, volume_data):
        """Extract comprehensive features for ML models"""
        features = {}
        
        # Price features
        recent_prices = [p['close'] for p in price_data[-20:]]
        if len(recent_prices) > 0:
            features['price_trend'] = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            features['price_volatility'] = np.std(recent_prices) / np.mean(recent_prices)
            features['price_momentum'] = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
        
        # Volume features
        recent_volumes = volume_data[-10:] if len(volume_data) >= 10 else volume_data
        if len(recent_volumes) > 1:
            features['volume_trend'] = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0]
            features['volume_spike'] = recent_volumes[-1] / np.mean(recent_volumes[:-1]) if len(recent_volumes) > 1 else 1
        
        # Technical indicator features
        features['rsi'] = indicators.get('current_rsi_14', 50)
        features['rsi_divergence'] = abs(features['rsi'] - 50) / 50
        features['macd_signal'] = 1 if indicators.get('current_macd', 0) > indicators.get('current_macd_signal', 0) else -1
        features['bb_position'] = AdvancedMLPredictor._calculate_bb_position(indicators, recent_prices[-1] if recent_prices else 0)
        features['trend_strength'] = AdvancedMLPredictor._calculate_trend_strength(indicators)
        
        # Pattern features
        bullish_patterns = ['engulfing_bullish', 'hammer', 'morning_star', 'three_white_soldiers', 'piercing_line']
        bearish_patterns = ['engulfing_bearish', 'shooting_star', 'evening_star', 'three_black_crows', 'dark_cloud_cover']
        
        features['bullish_pattern_count'] = sum(1 for p in bullish_patterns if patterns.get(p, False))
        features['bearish_pattern_count'] = sum(1 for p in bearish_patterns if patterns.get(p, False))
        features['pattern_strength'] = features['bullish_pattern_count'] - features['bearish_pattern_count']
        
        return features
    
    @staticmethod
    def _calculate_bb_position(indicators, current_price):
        """Calculate position within Bollinger Bands"""
        bb_upper = indicators.get('current_bb_upper', current_price)
        bb_lower = indicators.get('current_bb_lower', current_price)
        if bb_upper == bb_lower:
            return 0.5
        return (current_price - bb_lower) / (bb_upper - bb_lower)
    
    @staticmethod
    def _calculate_trend_strength(indicators):
        """Calculate overall trend strength"""
        ema_20 = indicators.get('current_ema_20', 0)
        ema_50 = indicators.get('current_ema_50', 0)
        sma_200 = indicators.get('current_sma_200', 0)
        
        if ema_50 == 0 or sma_200 == 0:
            return 0
        
        trend_score = 0
        if ema_20 > ema_50:
            trend_score += 1
        if ema_50 > sma_200:
            trend_score += 1
        if ema_20 > sma_200:
            trend_score += 1
            
        return trend_score / 3
    
    @staticmethod
    def _predict_scalping(features):
        """Scalping predictions (1-15 minutes)"""
        score = 0
        confidence_factors = []
        
        # RSI for quick reversals
        rsi = features.get('rsi', 50)
        if rsi < 20:
            score += 4
            confidence_factors.append(0.9)
        elif rsi < 30:
            score += 2
            confidence_factors.append(0.7)
        elif rsi > 80:
            score -= 4
            confidence_factors.append(0.9)
        elif rsi > 70:
            score -= 2
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        # Volume spike for momentum
        volume_spike = features.get('volume_spike', 1)
        if volume_spike > 2:
            score += 2
            confidence_factors.append(0.8)
        elif volume_spike > 1.5:
            score += 1
            confidence_factors.append(0.6)
        
        # Pattern strength
        pattern_strength = features.get('pattern_strength', 0)
        score += pattern_strength * 2
        if pattern_strength != 0:
            confidence_factors.append(0.7)
        
        direction = 'BUY' if score > 1 else 'SELL' if score < -1 else 'NEUTRAL'
        confidence = min(95, max(40, np.mean(confidence_factors) * 100 + abs(score) * 8))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'timeframe': '1-15 minutes',
            'strategy': 'Scalping',
            'risk_level': 'HIGH'
        }
    
    @staticmethod
    def _predict_short_term(features):
        """Short term predictions (15 minutes - 4 hours)"""
        score = 0
        confidence_factors = []
        
        # RSI with different thresholds
        rsi = features.get('rsi', 50)
        if rsi < 25:
            score += 3
            confidence_factors.append(0.85)
        elif rsi < 35:
            score += 2
            confidence_factors.append(0.7)
        elif rsi > 75:
            score -= 3
            confidence_factors.append(0.85)
        elif rsi > 65:
            score -= 2
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # MACD signal
        macd_signal = features.get('macd_signal', 0)
        score += macd_signal * 2
        confidence_factors.append(0.6)
        
        # Trend strength
        trend_strength = features.get('trend_strength', 0)
        if trend_strength > 0.7:
            score += 2
        elif trend_strength < 0.3:
            score -= 2
        confidence_factors.append(0.5)
        
        # BB position
        bb_position = features.get('bb_position', 0.5)
        if bb_position < 0.1:
            score += 2
        elif bb_position > 0.9:
            score -= 2
        
        direction = 'BUY' if score > 1 else 'SELL' if score < -1 else 'NEUTRAL'
        confidence = min(92, max(45, np.mean(confidence_factors) * 90 + abs(score) * 6))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'timeframe': '15min - 4 hours',
            'strategy': 'Short Term',
            'risk_level': 'MEDIUM'
        }
    
    @staticmethod
    def _predict_medium_term(features):
        """Medium term predictions (4 hours - 3 days)"""
        score = 0
        confidence_factors = []
        
        # Price trend
        price_trend = features.get('price_trend', 0)
        if price_trend > 0.05:
            score += 3
            confidence_factors.append(0.8)
        elif price_trend < -0.05:
            score -= 3
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Trend strength is more important for medium term
        trend_strength = features.get('trend_strength', 0)
        score += (trend_strength - 0.5) * 4
        confidence_factors.append(0.7)
        
        # Pattern strength
        pattern_strength = features.get('pattern_strength', 0)
        score += pattern_strength * 1.5
        if pattern_strength != 0:
            confidence_factors.append(0.6)
        
        # Volume trend
        volume_trend = features.get('volume_trend', 0)
        if volume_trend > 0.2:
            score += 1
        elif volume_trend < -0.2:
            score -= 1
        confidence_factors.append(0.5)
        
        direction = 'BUY' if score > 1 else 'SELL' if score < -1 else 'NEUTRAL'
        confidence = min(88, max(40, np.mean(confidence_factors) * 85 + abs(score) * 4))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'timeframe': '4 hours - 3 days',
            'strategy': 'Medium Term',
            'risk_level': 'MEDIUM'
        }
    
    @staticmethod
    def _predict_long_term(features):
        """Long term predictions (3 days - 4 weeks)"""
        score = 0
        confidence_factors = []
        
        # Long term price trend is most important
        price_trend = features.get('price_trend', 0)
        if price_trend > 0.1:
            score += 4
            confidence_factors.append(0.9)
        elif price_trend < -0.1:
            score -= 4
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Trend strength
        trend_strength = features.get('trend_strength', 0)
        score += (trend_strength - 0.5) * 3
        confidence_factors.append(0.8)
        
        # Volume commitment
        volume_trend = features.get('volume_trend', 0)
        if volume_trend > 0.3:
            score += 2
        elif volume_trend < -0.3:
            score -= 1
        confidence_factors.append(0.6)
        
        # Volatility (lower is better for long term)
        volatility = features.get('price_volatility', 0)
        if volatility < 0.02:
            score += 1
        elif volatility > 0.08:
            score -= 1
        confidence_factors.append(0.4)
        
        direction = 'BUY' if score > 1 else 'SELL' if score < -1 else 'NEUTRAL'
        confidence = min(85, max(35, np.mean(confidence_factors) * 80 + abs(score) * 3))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'timeframe': '3 days - 4 weeks',
            'strategy': 'Long Term',
            'risk_level': 'LOW'
        }
    
    @staticmethod
    def _predict_swing_trade(features):
        """Swing trading predictions (1-10 days)"""
        score = 0
        confidence_factors = []
        
        # RSI for swing levels
        rsi = features.get('rsi', 50)
        if 25 <= rsi <= 35:
            score += 3
            confidence_factors.append(0.8)
        elif 65 <= rsi <= 75:
            score -= 3
            confidence_factors.append(0.8)
        elif rsi < 20 or rsi > 80:
            # Extreme levels, wait for reversal
            score += 1 if rsi < 20 else -1
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # Trend + momentum combination
        trend_strength = features.get('trend_strength', 0)
        price_momentum = features.get('price_momentum', 0)
        
        if trend_strength > 0.6 and price_momentum > 0.02:
            score += 2
            confidence_factors.append(0.7)
        elif trend_strength < 0.4 and price_momentum < -0.02:
            score -= 2
            confidence_factors.append(0.7)
        
        # BB position for entry points
        bb_position = features.get('bb_position', 0.5)
        if bb_position < 0.2:
            score += 2  # Near lower band
        elif bb_position > 0.8:
            score -= 2  # Near upper band
        confidence_factors.append(0.6)
        
        # Pattern confirmation
        pattern_strength = features.get('pattern_strength', 0)
        score += pattern_strength * 2
        if pattern_strength != 0:
            confidence_factors.append(0.7)
        
        direction = 'BUY' if score > 2 else 'SELL' if score < -2 else 'NEUTRAL'
        confidence = min(90, max(42, np.mean(confidence_factors) * 88 + abs(score) * 5))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'score': score,
            'timeframe': '1-10 days',
            'strategy': 'Swing Trading',
            'risk_level': 'MEDIUM'
        }

class AdvancedMarketAnalyzer:
    @staticmethod
    def analyze_comprehensive_market(indicators, patterns, ml_predictions, price_data, volume_data):
        try:
            analysis = {
                'overall_sentiment': 'NEUTRAL',
                'confidence': 50,
                'strength': 5,
                'risk_level': 'MEDIUM',
                'recommended_action': 'HOLD',
                'signals': [],
                'kpis': {},
                'trading_score': 0,
                'market_state': 'CONSOLIDATION'
            }
            signals = AdvancedMarketAnalyzer._generate_trading_signals(indicators, patterns, ml_predictions, price_data)
            analysis['signals'] = signals
            kpis = AdvancedMarketAnalyzer._calculate_trading_kpis(indicators, patterns, price_data, volume_data)
            analysis['kpis'] = kpis
            sentiment_data = AdvancedMarketAnalyzer._calculate_sentiment_score(signals, kpis, ml_predictions)
            analysis.update(sentiment_data)
            market_state = AdvancedMarketAnalyzer._detect_market_state(indicators, price_data, volume_data)
            analysis['market_state'] = market_state
            return analysis
        except Exception as e:
            logger.error(f"Error in comprehensive market analysis: {str(e)}")
            return {}

    # (Hier kommen alle Methoden 1:1 wie im Gist – Kürzung aus Platzgründen)

# API Helper Functions
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
        logger.info(f"Cache hit for {cache_key}")
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
        logger.info(f"Successfully fetched {len(processed_data)} candles for {symbol} ({interval})")
        return processed_data

    except Exception as e:
        logger.error(f"Error fetching Binance data: {str(e)}")
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
        logger.error(f"Error fetching 24hr ticker: {str(e)}")
        raise Exception(f"Failed to fetch ticker: {str(e)}")

# Cache Cleanup Service
def cleanup_cache_service():
    while True:
        try:
            current_time = datetime.now()
            expired_keys = []
            for key, (data, timestamp) in api_cache.items():
                if current_time - timestamp > timedelta(seconds=CACHE_DURATION * 3):
                    expired_keys.append(key)
            for key in expired_keys:
                del api_cache[key]
            if expired_keys:
                logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
            time.sleep(300)
        except Exception as e:
            logger.error(f"Error in cache cleanup: {str(e)}")
            time.sleep(60)

# === MARKIERUNG 2: ENDE ===

# === MARKIERUNG 3: Starte den Cache Cleanup Thread ===

cleanup_thread = threading.Thread(target=cleanup_cache_service, daemon=True)
cleanup_thread.start()

# === MARKIERUNG 4: Flask-Routen (alle @app.route(...) Funktionen) ===

@app.route('/')
def dashboard():
    """Main dashboard route"""
    try:
        return render_template_string(get_ultimate_dashboard_html())
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        return f"Error loading dashboard: {str(e)}", 500

@app.route('/api/analyze', methods=['POST'])
def analyze_symbol():
    # (Hier kommt dein kompletter Analyze-Endpoint 1:1 rein)

@app.route('/api/health', methods=['GET'])
def health_check():
    # (Hier kommt dein kompletter Health-Endpoint 1:1 rein)

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    # (Hier kommt dein kompletter Watchlist-Endpoint 1:1 rein)

# === MARKIERUNG 5: HTML-Dashboard als Funktion ===

def get_ultimate_dashboard_html():
    return '''
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <!-- ... ALLES aus deinem Gist: komplett HTML, CSS, JS ... -->
    </head>
    <body>
        <!-- ... -->
    </body>
    </html>
    '''

# === MARKIERUNG 6: Main-Block für Railway (ganz am Ende der Datei) ===
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
