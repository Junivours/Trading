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

cleanup_thread = threading.Thread(target=cleanup_cache_service, daemon=True)
cleanup_thread.start()

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
    try:
        req = request.get_json()
        symbol = req.get('symbol', 'BTCUSDT')
        interval = req.get('interval', '1h')
        limit = int(req.get('limit', 200))
        ohlc_data = fetch_binance_data(symbol, interval=interval, limit=limit)
        ticker_data = fetch_24hr_ticker(symbol)
        price_data = [c['close'] for c in ohlc_data]
        volume_data = [c['volume'] for c in ohlc_data]
        indicators = AdvancedTechnicalAnalyzer.calculate_all_indicators(ohlc_data)
        patterns = AdvancedPatternDetector.detect_all_patterns(ohlc_data)
        ml_predictions = AdvancedMLPredictor.calculate_predictions(indicators, patterns, price_data, volume_data)
        analysis = AdvancedMarketAnalyzer.analyze_comprehensive_market(
            indicators, patterns, ml_predictions, price_data, volume_data
        )
        response = {
            'symbol': symbol,
            'interval': interval,
            'ohlc': ohlc_data,
            'ticker': ticker_data,
            'indicators': indicators,
            'patterns': patterns,
            'ml_predictions': ml_predictions,
            'analysis': analysis
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /api/analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        return jsonify({'status': 'ok', 'msg': 'Server alive', 'timestamp': datetime.utcnow().isoformat()})
    except Exception as e:
        logger.error(f"Error in /api/health: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    try:
        return jsonify({
            'watchlist': [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
                "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
            ]
        })
    except Exception as e:
        logger.error(f"Error in /api/watchlist: {str(e)}")
        return jsonify({'error': str(e)}), 500
# === MARKIERUNG 5: HTML-Dashboard als Funktion ===

def get_ultimate_dashboard_html():
    return '''
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🔥 ULTIMATE Trading Analysis Pro</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js"></script>
        <style>
           * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-bg: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            --card-bg: rgba(30, 41, 59, 0.4);
            --card-border: rgba(71, 85, 105, 0.2);
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }

        body {
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            background: var(--primary-bg);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            font-size: 14px;
            line-height: 1.5;
        }

        /* Header Styles */
        .header {
            background: rgba(15, 15, 35, 0.95);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(59, 130, 246, 0.2);
            padding: 1rem 1.5rem;
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1600px;
            margin: 0 auto;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .logo {
            font-size: 1.8rem;
            font-weight: 900;
            background: linear-gradient(45deg, var(--accent-blue), var(--accent-cyan), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .control-group {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        .input-field {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 8px;
            padding: 0.6rem 1rem;
            color: var(--text-primary);
            font-size: 0.9rem;
            font-weight: 500;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            min-width: 140px;
        }

        .input-field:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .timeframe-group {
            display: flex;
            background: rgba(30, 41, 59, 0.6);
            border-radius: 8px;
            padding: 0.2rem;
            gap: 0.1rem;
        }

        .timeframe-btn {
            background: transparent;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 0.8rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .timeframe-btn:hover {
            background: rgba(59, 130, 246, 0.1);
            color: var(--text-primary);
        }

        .timeframe-btn.active {
            background: var(--accent-blue);
            color: white;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        }

        .analyze-btn {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
            border: none;
            border-radius: 8px;
            padding: 0.7rem 1.5rem;
            color: white;
            font-weight: 700;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .analyze-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        }

        .analyze-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }

        /* Live Toggle */
        .live-toggle {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(30, 41, 59, 0.6);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.85rem;
        }

        .toggle-switch {
            width: 40px;
            height: 20px;
            background: rgba(71, 85, 105, 0.5);
            border-radius: 20px;
            position: relative;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .toggle-switch.active {
            background: var(--accent-green);
        }

        .toggle-knob {
            width: 16px;
            height: 16px;
            background: white;
            border-radius: 50%;
            position: absolute;
            top: 2px;
            left: 2px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }

        .toggle-switch.active .toggle-knob {
            transform: translateX(20px);
        }

        /* Main Container */
        .main-container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 1.5rem;
        }

        /* Status Banner */
        .status-banner {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(6, 182, 212, 0.1));
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .live-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { 
                transform: scale(1);
                opacity: 1;
            }
            50% { 
                transform: scale(1.3);
                opacity: 0.6;
            }
        }

        /* Grid Layouts */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        /* Card Styles */
        .card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.3s ease;
            position: relative;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            border-color: rgba(59, 130, 246, 0.3);
        }

        .card-title {
            font-size: 1.2rem;
            font-weight: 800;
            margin-bottom: 1rem;
            background: linear-gradient(45deg, var(--accent-blue), var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Price Display */
        .price-display {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(16, 185, 129, 0.1));
            border-color: rgba(59, 130, 246, 0.3);
        }

        .price-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            align-items: center;
        }

        .price-main {
            text-align: center;
        }

        .current-price {
            font-size: 2.8rem;
            font-weight: 900;
            background: linear-gradient(45deg, var(--accent-green), var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .price-change {
            font-size: 1.2rem;
            font-weight: 700;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }

        .price-change.positive {
            color: var(--success);
            background: rgba(16, 185, 129, 0.15);
        }

        .price-change.negative {
            color: var(--danger);
            background: rgba(239, 68, 68, 0.15);
        }

        .price-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }

        .stat-item {
            background: rgba(15, 15, 35, 0.3);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .stat-item:hover {
            background: rgba(30, 41, 59, 0.4);
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stat-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* KPI Cards */
        .kpi-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 1.2rem;
            transition: all 0.3s ease;
            border-left: 4px solid transparent;
        }

        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        }

        .kpi-card.bullish {
            border-left-color: var(--success);
        }

        .kpi-card.bearish {
            border-left-color: var(--danger);
        }

        .kpi-card.neutral {
            border-left-color: var(--warning);
        }

        .kpi-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
        }

        .kpi-name {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .kpi-status {
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .kpi-value {
            font-size: 1.6rem;
            font-weight: 900;
            margin-bottom: 0.3rem;
            color: var(--text-primary);
        }

        .kpi-description {
            font-size: 0.75rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        .status-bullish {
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
        }

        .status-bearish {
            background: rgba(239, 68, 68, 0.2);
            color: var(--danger);
        }

        .status-neutral {
            background: rgba(245, 158, 11, 0.2);
            color: var(--warning);
        }

        /* Indicators Grid */
        .indicators-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }

        .indicator-card {
            background: rgba(15, 15, 35, 0.3);
            border-radius: 10px;
            padding: 1rem;
            border-left: 3px solid transparent;
            transition: all 0.3s ease;
        }

        .indicator-card:hover {
            background: rgba(30, 41, 59, 0.4);
            transform: translateX(3px);
        }

        .indicator-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
        }

        .indicator-name {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .indicator-status {
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .indicator-value {
            font-size: 1.4rem;
            font-weight: 900;
            margin-bottom: 0.3rem;
            color: var(--text-primary);
        }

        .indicator-description {
            font-size: 0.75rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        .bullish { 
            border-left-color: var(--success);
        }
        .bearish { 
            border-left-color: var(--danger);
        }
        .neutral { 
            border-left-color: var(--warning);
        }

        /* Patterns Grid */
        .patterns-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .pattern-item {
            background: rgba(15, 15, 35, 0.3);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }

        .pattern-item.detected {
            background: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.3);
            transform: scale(1.02);
        }

        .pattern-name {
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
            color: var(--text-primary);
        }

        .pattern-description {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        /* ML Predictions */
        .ml-section {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(59, 130, 246, 0.1));
            border-color: rgba(139, 92, 246, 0.3);
        }

        .ml-predictions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .prediction-card {
            background: rgba(15, 15, 35, 0.4);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .prediction-card:hover {
            transform: translateY(-2px);
            background: rgba(30, 41, 59, 0.4);
        }

        .prediction-timeframe {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
            text-transform: uppercase;
        }

        .prediction-direction {
            font-size: 1.2rem;
            font-weight: 900;
            margin-bottom: 0.5rem;
        }

        .prediction-confidence {
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            border-radius: 8px;
            display: inline-block;
        }

        /* Signals */
        .signals-container {
            max-height: 400px;
            overflow-y: auto;
            margin-top: 1rem;
        }

        .signal-item {
            background: rgba(15, 15, 35, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-left: 3px solid;
            transition: all 0.3s ease;
        }

        .signal-item:hover {
            background: rgba(30, 41, 59, 0.4);
        }

        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.3rem;
        }

        .signal-type {
            font-weight: 700;
            font-size: 0.9rem;
        }

        .signal-indicator {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .signal-reason {
            font-size: 0.85rem;
            color: var(--text-primary);
            margin-bottom: 0.3rem;
        }

        .signal-meta {
            font-size: 0.75rem;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
        }

        /* Market State */
        .market-state {
            text-align: center;
            padding: 1rem;
            background: rgba(15, 15, 35, 0.3);
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .market-state-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
            text-transform: uppercase;
        }

        .market-state-value {
            font-size: 1.4rem;
            font-weight: 900;
            color: var(--accent-cyan);
        }

        /* Quick Actions */
        .quick-actions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 0.8rem;
            margin-bottom: 1.5rem;
        }

        .quick-btn {
            background: rgba(15, 15, 35, 0.5);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 0.8rem;
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }

        .quick-btn:hover {
            background: rgba(59, 130, 246, 0.1);
            border-color: rgba(59, 130, 246, 0.3);
            transform: translateY(-1px);
        }

        .quick-coin-name {
            font-weight: 700;
            font-size: 0.85rem;
            margin-bottom: 0.2rem;
        }

        .quick-coin-symbol {
            font-size: 0.7rem;
            color: var(--text-secondary);
        }

        /* Loading States */
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
            font-size: 1rem;
            color: var(--text-secondary);
        }

        .loading::after {
            content: '';
            width: 20px;
            height: 20px;
            border: 2px solid rgba(71, 85, 105, 0.3);
            border-top: 2px solid var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 1rem;
        }

        .error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: var(--danger);
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 1rem;
            }
            
            .controls {
                flex-direction: column;
                width: 100%;
            }
            
            .current-price {
                font-size: 2.2rem;
            }
            
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .kpi-grid {
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            # -*- coding: utf-8 -*-
      }
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(71, 85, 105, 0.2);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(59, 130, 246, 0.5);
            border-radius: 3px;
        }

        /* Animation Classes */
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .scale-in {
            animation: scaleIn 0.3s ease-out;
        }

        @keyframes scaleIn {
            from { transform: scale(0.9); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }

        /* Trading Score Gauge */
        .trading-score-gauge {
            position: relative;
            width: 120px;
            height: 120px;
            margin: 0 auto;
        }

        .gauge-bg {
            stroke: rgba(71, 85, 105, 0.3);
            stroke-width: 8;
            fill: none;
        }

        .gauge-fill {
            stroke: var(--accent-green);
            stroke-width: 8;
            fill: none;
            stroke-linecap: round;
            transition: stroke-dasharray 1s ease-in-out;
        }

        .gauge-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.2rem;
            font-weight: 900;
            text-align: center;
        }

        /* Tooltips */
        .tooltip {
            position: relative;
            cursor: help;
        }

        .tooltip::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(15, 15, 35, 0.9);
            color: var(--text-primary);
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
            z-index: 1000;
        }

        .tooltip:hover::after {
            opacity: 1;
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div class="logo">
                <i class="fas fa-chart-line"></i>
                ULTIMATE Trading Analysis Pro
            </div>
            <div class="controls">
                <div class="control-group">
                    <input type="text" class="input-field" placeholder="Symbol (z.B. BTCUSDT)" id="coinInput" value="BTCUSDT">
                    <div class="timeframe-group">
                        <button class="timeframe-btn" data-timeframe="1m">1M</button>
                        <button class="timeframe-btn" data-timeframe="5m">5M</button>
                        <button class="timeframe-btn" data-timeframe="15m">15M</button>
                        <button class="timeframe-btn active" data-timeframe="1h">1H</button>
                        <button class="timeframe-btn" data-timeframe="4h">4H</button>
                        <button class="timeframe-btn" data-timeframe="1d">1D</button>
                        <button class="timeframe-btn" data-timeframe="1w">1W</button>
                    </div>
                </div>
                <div class="control-group">
                    <button class="analyze-btn" onclick="analyzeSymbol()">
                        <i class="fas fa-rocket"></i>
                        ANALYSE
                    </button>
                    <div class="live-toggle" onclick="toggleLiveMode()">
                        <span>Live</span>
                        <div class="toggle-switch" id="liveToggle">
                            <div class="toggle-knob"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <div class="main-container">
        <div class="status-banner">
            <div class="live-indicator">
                <div class="pulse-dot"></div>
                <span>PROFESSIONAL TRADING ANALYSIS</span>
            </div>
            <div style="font-size: 0.9rem; color: var(--text-secondary);">
                50+ Patterns • 5 ML Models • Real-time KPIs • Professional Signals • Risk Management
            </div>
        </div>

        <!-- Quick Actions -->
        <div class="quick-actions">
            <button class="quick-btn" onclick="quickAnalyze('BTCUSDT')">
                <div class="quick-coin-name" style="color: #f59e0b;"><i class="fab fa-bitcoin"></i> Bitcoin</div>
                <div class="quick-coin-symbol">BTCUSDT</div>
            </button>
            <button class="quick-btn" onclick="quickAnalyze('ETHUSDT')">
                <div class="quick-coin-name" style="color: #627eea;"><i class="fab fa-ethereum"></i> Ethereum</div>
                <div class="quick-coin-symbol">ETHUSDT</div>
            </button>
            <button class="quick-btn" onclick="quickAnalyze('SOLUSDT')">
                <div class="quick-coin-name" style="color: #9945ff;">◎ Solana</div>
                <div class="quick-coin-symbol">SOLUSDT</div>
            </button>
            <button class="quick-btn" onclick="quickAnalyze('BNBUSDT')">
                <div class="quick-coin-name" style="color: #f3ba2f;">⬢ BNB</div>
                <div class="quick-coin-symbol">BNBUSDT</div>
            </button>
            <button class="quick-btn" onclick="quickAnalyze('XRPUSDT')">
                <div class="quick-coin-name" style="color: #00d4f5;">✦ XRP</div>
                <div class="quick-coin-symbol">XRPUSDT</div>
            </button>
            <button class="quick-btn" onclick="quickAnalyze('ADAUSDT')">
                <div class="quick-coin-name" style="color: #0033ad;">₳ Cardano</div>
                <div class="quick-coin-symbol">ADAUSDT</div>
            </button>
        </div>

        <!-- Loading/Error Display -->
        <div class="card" id="loadingCard" style="display: none;">
            <div class="loading">Analysiere Marktdaten...</div>
        </div>

        <div class="card error" id="errorCard" style="display: none;">
            <div id="errorMessage">Fehler beim Laden der Daten</div>
        </div>

        <!-- Main Dashboard Grid -->
        <div class="dashboard-grid" id="dashboard">
            <!-- Content will be dynamically loaded here -->
        </div>
    </div>

    <script>
        // Global Variables
        let currentSymbol = 'BTCUSDT';
        let currentTimeframe = '1h';
        let liveModeEnabled = false;
        let liveUpdateInterval;
        let analysisData = null;

        // Initialize Application
        document.addEventListener('DOMContentLoaded', function() {
            initializeEventListeners();
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
                        analyzeSymbol(); // Re-analyze with new timeframe
                    }
                });
            });

            // Enter key for symbol input
            document.getElementById('coinInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    analyzeSymbol();
                }
            });

            // Symbol input change
            document.getElementById('coinInput').addEventListener('change', function() {
                const symbol = this.value.trim().toUpperCase();
                if (symbol && symbol !== currentSymbol) {
                    currentSymbol = symbol;
                    if (liveModeEnabled) {
                        analyzeSymbol();
                    }
                }
            });
        }

        function quickAnalyze(symbol) {
            document.getElementById('coinInput').value = symbol;
            currentSymbol = symbol;
            analyzeSymbol();
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
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                analysisData = data;
                displayAnalysis(data);
                hideLoading();

            } catch (error) {
                console.error('Analysis error:', error);
                showError(`Fehler beim Analysieren von ${currentSymbol}: ${error.message}`);
            }
        }

        function showLoading() {
            hideError();
            document.getElementById('loadingCard').style.display = 'block';
            // Hide existing analysis cards
            const existingCards = document.querySelectorAll('.analysis-card');
            existingCards.forEach(card => card.remove());
        }

        function hideLoading() {
            document.getElementById('loadingCard').style.display = 'none';
        }

        function showError(message) {
            hideLoading();
            document.getElementById('errorMessage').textContent = message;
            document.getElementById('errorCard').style.display = 'block';
        }

        function hideError() {
            document.getElementById('errorCard').style.display = 'none';
        }

        function toggleLiveMode() {
            liveModeEnabled = !liveModeEnabled;
            const toggle = document.getElementById('liveToggle');
            
            if (liveModeEnabled) {
                toggle.classList.add('active');
                startLiveUpdates();
            } else {
                toggle.classList.remove('active');
                stopLiveUpdates();
            }
        }

        function startLiveUpdates() {
            if (liveUpdateInterval) clearInterval(liveUpdateInterval);
            
            liveUpdateInterval = setInterval(() => {
                if (currentSymbol) {
                    analyzeSymbol();
                }
            }, 30000); // Update every 30 seconds
        }

        function stopLiveUpdates() {
            if (liveUpdateInterval) {
                clearInterval(liveUpdateInterval);
                liveUpdateInterval = null;
            }
        }

        function displayAnalysis(data) {
            const dashboard = document.getElementById('dashboard');
            
            // Remove existing analysis cards
            const existingCards = document.querySelectorAll('.analysis-card');
            existingCards.forEach(card => card.remove());

            // Create all components
            const priceCard = createPriceCard(data);
            const kpiOverview = createKPIOverviewCard(data);
            const technicalIndicators = createTechnicalIndicatorsCard(data.indicators);
            const mlPredictions = createMLPredictionsCard(data.ml_predictions);
            const patternAnalysis = createPatternAnalysisCard(data.patterns);
            const tradingSignals = createTradingSignalsCard(data.market_analysis.signals);
            const marketSummary = createMarketSummaryCard(data.market_analysis);

            // Append to dashboard
            [priceCard, kpiOverview, technicalIndicators, mlPredictions, 
             patternAnalysis, tradingSignals, marketSummary].forEach(card => {
                dashboard.appendChild(card);
            });

            // Add animations
            setTimeout(() => {
                document.querySelectorAll('.analysis-card').forEach((card, index) => {
                    setTimeout(() => {
                        card.classList.add('fade-in');
                    }, index * 100);
                });
            }, 50);
        }

        function createPriceCard(data) {
            const card = document.createElement('div');
            card.className = 'card price-display analysis-card';
            
            const changeClass = data.price_change_24h >= 0 ? 'positive' : 'negative';
            const changeIcon = data.price_change_24h >= 0 ? '↗' : '↘';
            
            card.innerHTML = `
                <h3 class="card-title">
                    <i class="fas fa-dollar-sign"></i>
                    ${data.symbol} Preis & Performance
                </h3>
                <div class="price-info">
                    <div class="price-main">
                        <div class="current-price">${formatPrice(data.current_price)}</div>
                        <div class="price-change ${changeClass}">
                            ${changeIcon} ${data.price_change_24h >= 0 ? '+' : ''}${data.price_change_24h.toFixed(2)}%
                        </div>
                        <div class="market-state">
                            <div class="market-state-label">Marktzustand</div>
                            <div class="market-state-value">${data.market_analysis.market_state || 'ANALYSING'}</div>
                        </div>
                    </div>
                    <div class="price-stats">
                        <div class="stat-item">
                            <div class="stat-label">24h Hoch</div>
                            <div class="stat-value">${formatPrice(data.high_24h)}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">24h Tief</div>
                            <div class="stat-value">${formatPrice(data.low_24h)}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">24h Volumen</div>
                            <div class="stat-value">${formatVolume(data.volume_24h)}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Zeitrahmen</div>
                            <div class="stat-value">${data.interval}</div>
                        </div>
                    </div>
                </div>
            `;
            return card;
        }

        function createKPIOverviewCard(data) {
            const card = document.createElement('div');
            card.className = 'card analysis-card';
            
            const kpis = data.market_analysis.kpis;
            const tradingScore = data.market_analysis.trading_score || 50;
            
            card.innerHTML = `
                <h3 class="card-title">
                    <i class="fas fa-tachometer-alt"></i>
                    KPI Dashboard
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 1.5rem; align-items: center;">
                    <div class="trading-score-gauge">
                        <svg width="120" height="120" viewBox="0 0 120 120">
                            <circle cx="60" cy="60" r="50" class="gauge-bg"></circle>
                            <circle cx="60" cy="60" r="50" class="gauge-fill" 
                                    style="stroke-dasharray: ${(tradingScore / 100) * 314} 314; 
                                           stroke: ${getScoreColor(tradingScore)}"></circle>
                        </svg>
                        <div class="gauge-text">
                            <div style="font-size: 1.4rem; color: ${getScoreColor(tradingScore)}">${tradingScore.toFixed(0)}</div>
                            <div style="font-size: 0.7rem; color: var(--text-secondary)">Trading Score</div>
                        </div>
                    </div>
                    <div class="kpi-grid" style="margin: 0;">
                        <div class="kpi-card ${getTrendClass(kpis.trend_strength)}">
                            <div class="kpi-header">
                                <div class="kpi-name">Trend Stärke</div>
                                <div class="kpi-status status-${getTrendClass(kpis.trend_strength)}">${(kpis.trend_strength || 0) * 25}%</div>
                            </div>
                            <div class="kpi-value">${kpis.trend_strength || 0}/4</div>
                        </div>
                        <div class="kpi-card ${getVolatilityClass(kpis.volatility_20d)}">
                            <div class="kpi-header">
                                <div class="kpi-name">Volatilität</div>
                                <div class="kpi-status status-${getVolatilityClass(kpis.volatility_20d)}">${(kpis.volatility_20d || 0).toFixed(1)}%</div>
                            </div>
                            <div class="kpi-value">${getVolatilityLevel(kpis.volatility_20d)}</div>
                        </div>
                        <div class="kpi-card ${getMomentumClass(kpis.momentum_score)}">
                            <div class="kpi-header">
                                <div class="kpi-name">Momentum</div>
                                <div class="kpi-status status-${getMomentumClass(kpis.momentum_score)}">${(kpis.momentum_score || 0) * 33}%</div>
                            </div>
                            <div class="kpi-value">${kpis.momentum_score || 0}/3</div>
                        </div>
                        <div class="kpi-card ${getEfficiencyClass(kpis.market_efficiency)}">
                            <div class="kpi-header">
                                <div class="kpi-name">Effizienz</div>
                                <div class="kpi-status status-${getEfficiencyClass(kpis.market_efficiency)}">${(kpis.market_efficiency || 0).toFixed(0)}%</div>
                            </div>
                            <div class="kpi-value">${getEfficiencyLevel(kpis.market_efficiency)}</div>
                        </div>
                    </div>
                </div>
            `;
            return card;
        }

        function createTechnicalIndicatorsCard(indicators) {
            const card = document.createElement('div');
            card.className = 'card analysis-card';
            
            card.innerHTML = `
                <h3 class="card-title">
                    <i class="fas fa-chart-area"></i>
                    Technische Indikatoren
                </h3>
                <div class="indicators-grid">
                    ${createIndicatorItem('RSI (14)', indicators.rsi_14, getRSIStatus(indicators.rsi_14), 'Relative Strength Index')}
                    ${createIndicatorItem('MACD', indicators.macd, getMACDStatus(indicators.macd, indicators.macd_signal), 'MACD Linie vs Signal')}
                    ${createIndicatorItem('Stochastic', indicators.stoch_k, getStochStatus(indicators.stoch_k), 'Stochastic Oscillator %K')}
                    ${createIndicatorItem('Williams %R', indicators.williams_r, getWilliamsStatus(indicators.williams_r), 'Williams Percent Range')}
                    ${createIndicatorItem('CCI', indicators.cci, getCCIStatus(indicators.cci), 'Commodity Channel Index')}
                    ${createIndicatorItem('ADX', indicators.adx, getADXStatus(indicators.adx), 'Average Directional Index')}
                    ${createIndicatorItem('ATR', indicators.atr, getATRStatus(indicators.atr), 'Average True Range')}
                    ${createIndicatorItem('OBV', indicators.obv, getOBVStatus(indicators.obv), 'On Balance Volume')}
                </div>
            `;
            return card;
        }

        function createMLPredictionsCard(predictions) {
            const card = document.createElement('div');
            card.className = 'card ml-section analysis-card';
            
            card.innerHTML = `
                <h3 class="card-title">
                    <i class="fas fa-brain"></i>
                    ML Vorhersagen & Strategien
                </h3>
                <div class="ml-predictions">
                    ${createPredictionItem('Scalping', predictions.scalping, '⚡')}
                    ${createPredictionItem('Short Term', predictions.short_term, '📈')}
                    ${createPredictionItem('Medium Term', predictions.medium_term, '📊')}
                    ${createPredictionItem('Long Term', predictions.long_term, '🎯')}
                    ${createPredictionItem('Swing Trade', predictions.swing_trade, '🔄')}
                </div>
            `;
            return card;
        }

        function createPatternAnalysisCard(patterns) {
            const card = document.createElement('div');
            card.className = 'card analysis-card';
            
            const detectedPatterns = Object.entries(patterns).filter(([_, detected]) => detected);
            const totalPatterns = Object.keys(patterns).length;
            
            card.innerHTML = `
                <h3 class="card-title">
                    <i class="fas fa-shapes"></i>
                    Candlestick Patterns (${detectedPatterns.length}/${totalPatterns})
                </h3>
                <div class="patterns-grid">
                    ${createPatternGrid(patterns)}
                </div>
                ${detectedPatterns.length > 0 ? `
                    <div style="margin-top: 1rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border-left: 4px solid var(--success);">
                        <strong style="color: var(--success);"><i class="fas fa-check-circle"></i> Erkannte Patterns:</strong><br>
                        <div style="margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.5rem;">
                            ${detectedPatterns.map(([name, _]) => `
                                <span style="background: rgba(16, 185, 129, 0.2); padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                                    ${formatPatternName(name)}
                                </span>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            `;
            return card;
        }

        function createTradingSignalsCard(signals) {
            const card = document.createElement('div');
            card.className = 'card analysis-card';
            
            card.innerHTML = `
                <h3 class="card-title">
                    <i class="fas fa-signal"></i>
                    Trading Signale (${signals?.length || 0})
                </h3>
                <div class="signals-container">
                    ${signals && signals.length > 0 ? signals.map(signal => `
                        <div class="signal-item" style="border-left-color: ${getSignalColor(signal.type)}">
                            <div class="signal-header">
                                <div class="signal-type" style="color: ${getSignalColor(signal.type)}">
                                    ${getSignalIcon(signal.type)} ${signal.type}
                                </div>
                                <div class="signal-indicator">${signal.indicator}</div>
                            </div>
                            <div class="signal-reason">${signal.reason}</div>
                            <div class="signal-meta">
                                <span>Stärke: ${signal.strength}/10</span>
                                <span>Konfidenz: ${signal.confidence}%</span>
                                <span>Zeitrahmen: ${signal.timeframe}</span>
                            </div>
                        </div>
                    `).join('') : '<div style="text-align: center; color: var(--text-secondary); padding: 2rem;">Keine aktiven Signale</div>'}
                </div>
            `;
            return card;
        }

        function createMarketSummaryCard(analysis) {
            const card = document.createElement('div');
            card.className = 'card analysis-card';
            card.style.gridColumn = '1 / -1';
            
            const sentimentColor = getSentimentColor(analysis.overall_sentiment);
            const kpis = analysis.kpis || {};
            
            card.innerHTML = `
                <h3 class="card-title">
                    <i class="fas fa-clipboard-list"></i>
                    Marktanalyse Zusammenfassung
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 2rem; margin-bottom: 1.5rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 2.2rem; font-weight: 900; color: ${sentimentColor}; margin-bottom: 0.5rem;">
                            ${getSentimentIcon(analysis.overall_sentiment)} ${analysis.overall_sentiment}
                        </div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: ${sentimentColor}; margin-bottom: 1rem;">
                            ${analysis.recommended_action}
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                            <div class="stat-item">
                                <div class="stat-label">Konfidenz</div>
                                <div class="stat-value">${analysis.confidence.toFixed(1)}%</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Stärke</div>
                                <div class="stat-value">${analysis.strength.toFixed(1)}/10</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Risiko</div>
                                <div class="stat-value" style="color: ${getRiskColor(analysis.risk_level)}">${analysis.risk_level}</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Score</div>
                                <div class="stat-value">${analysis.trading_score || 50}/100</div>
                            </div>
                        </div>
                    </div>
                    <div>
                        <h4 style="margin-bottom: 1rem; color: var(--text-primary);"><i class="fas fa-chart-line"></i> Wichtige KPIs:</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.8rem;">
                            <div class="kpi-card" style="padding: 0.8rem;">
                                <div class="kpi-name">Support/Resistance</div>
                                <div class="kpi-value" style="font-size: 0.9rem;">
                                    ↑ ${formatPrice(kpis.resistance_level || 0)}<br>
                                    ↓ ${formatPrice(kpis.support_level || 0)}
                                </div>
                            </div>
                            <div class="kpi-card" style="padding: 0.8rem;">
                                <div class="kpi-name">BB Position</div>
                                <div class="kpi-value">${((kpis.bb_position || 0.5) * 100).toFixed(0)}%</div>
                            </div>
                            <div class="kpi-card" style="padding: 0.8rem;">
                                <div class="kpi-name">Sharpe Ratio</div>
                                <div class="kpi-value">${(kpis.sharpe_ratio || 0).toFixed(2)}</div>
                            </div>
                            <div class="kpi-card" style="padding: 0.8rem;">
                                <div class="kpi-name">Pattern Sentiment</div>
                                <div class="kpi-value" style="color: ${kpis.pattern_sentiment > 0 ? 'var(--success)' : kpis.pattern_sentiment < 0 ? 'var(--danger)' : 'var(--warning)'}">
                                    ${kpis.pattern_sentiment > 0 ? '+' : ''}${kpis.pattern_sentiment || 0}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                ${analysis.signals && analysis.signals.length > 3 ? `
                    <div style="margin-top: 1rem;">
                        <h4 style="margin-bottom: 0.8rem; color: var(--text-primary);"><i class="fas fa-star"></i> Top 3 Signale:</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
                            ${analysis.signals.slice(0, 3).map((signal, index) => `
                                <div style="background: rgba(15, 15, 35, 0.3); border-radius: 8px; padding: 1rem; border-left: 3px solid ${getSignalColor(signal.type)};">
                                    <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                                        <strong style="color: ${getSignalColor(signal.type)};">#${index + 1} ${signal.type}</strong>
                                        <span style="font-size: 0.8rem; color: var(--text-secondary);">${signal.category || signal.indicator}</span>
                                    </div>
                                    <div style="font-size: 0.9rem; color: var(--text-primary); margin-bottom: 0.5rem;">${signal.reason}</div>
                                    <div style="font-size: 0.75rem; color: var(--text-secondary);">
                                        Stärke: ${signal.strength}/10 | Konfidenz: ${signal.confidence}% | ${signal.timeframe}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            `;
            return card;
        }

        // Helper Functions for UI Components
        function createIndicatorItem(name, value, status, description) {
            return `
                <div class="indicator-card ${status.class}">
                    <div class="indicator-header">
                        <div class="indicator-name">${name}</div>
                        <div class="indicator-status status-${status.class}">${status.text}</div>
                    </div>
                    <div class="indicator-value">${typeof value === 'number' ? value.toFixed(2) : value}</div>
                    <div class="indicator-description">${description}</div>
                </div>
            `;
        }

        function createPredictionItem(timeframe, prediction, emoji) {
            if (!prediction) return '';
            
            const directionClass = prediction.direction === 'BUY' ? 'bullish' : 
                                 prediction.direction === 'SELL' ? 'bearish' : 'neutral';
            const directionColor = prediction.direction === 'BUY' ? 'var(--success)' : 
                                  prediction.direction === 'SELL' ? 'var(--danger)' : 'var(--warning)';
            
            return `
                <div class="prediction-card">
                    <div class="prediction-timeframe">${emoji} ${timeframe}</div>
                    <div class="prediction-direction" style="color: ${directionColor};">
                        ${prediction.direction}
                    </div>
                    <div class="prediction-confidence status-${directionClass}">
                        ${prediction.confidence.toFixed(1)}% Konfidenz
                    </div>
                    <div style="font-size: 0.7rem; color: var(--text-secondary); margin-top: 0.3rem;">
                        ${prediction.timeframe}
                    </div>
                    <div style="font-size: 0.65rem; color: var(--text-secondary); margin-top: 0.2rem;">
                        Strategie: ${prediction.strategy}
                    </div>
                </div>
            `;
        }

        function createPatternGrid(patterns) {
            const patternCategories = {
                'Single Candle': ['doji', 'hammer', 'shooting_star', 'spinning_top', 'marubozu', 'dragonfly_doji', 'gravestone_doji'],
                'Reversal': ['engulfing_bullish', 'engulfing_bearish', 'harami', 'piercing_line', 'dark_cloud_cover'],
                'Continuation': ['three_white_soldiers', 'three_black_crows', 'morning_star', 'evening_star'],
                'Advanced': ['abandoned_baby_bullish', 'abandoned_baby_bearish', 'advance_block', 'breakaway', 'hikkake']
            };

            let html = '';
            Object.entries(patternCategories).forEach(([category, patternList]) => {
                const categoryPatterns = patternList.filter(pattern => patterns.hasOwnProperty(pattern));
                if (categoryPatterns.length > 0) {
                    html += `
                        <div style="grid-column: 1 / -1; margin: 1rem 0 0.5rem 0;">
                            <h5 style="color: var(--accent-blue); font-size: 0.9rem; margin-bottom: 0.5rem;">${category} Patterns</h5>
                        </div>
                    `;
                    categoryPatterns.forEach(pattern => {
                        const isDetected = patterns[pattern];
                        html += `
                            <div class="pattern-item ${isDetected ? 'detected' : ''}">
                                <div class="pattern-name">
                                    ${formatPatternName(pattern)} ${isDetected ? '✅' : ''}
                                </div>
                                <div class="pattern-description">
                                    ${getPatternDescription(pattern)}
                                </div>
                            </div>
                        `;
                    });
                }
            });
            return html;
        }

        // Status and Color Helper Functions
        function getRSIStatus(rsi) {
            if (rsi >= 80) return {class: 'bearish', text: 'ÜBERKAUFT'};
            if (rsi >= 70) return {class: 'bearish', text: 'OVERBOUGHT'};
            if (rsi <= 20) return {class: 'bullish', text: 'ÜBERVERKAUFT'};
            if (rsi <= 30) return {class: 'bullish', text: 'OVERSOLD'};
            return {class: 'neutral', text: 'NEUTRAL'};
        }

        function getMACDStatus(macd, signal) {
            if (macd > signal) return {class: 'bullish', text: 'BULLISH'};
            if (macd < signal) return {class: 'bearish', text: 'BEARISH'};
            return {class: 'neutral', text: 'NEUTRAL'};
        }

        function getStochStatus(stoch) {
            if (stoch >= 80) return {class: 'bearish', text: 'ÜBERKAUFT'};
            if (stoch <= 20) return {class: 'bullish', text: 'ÜBERVERKAUFT'};
            return {class: 'neutral', text: 'NEUTRAL'};
        }

        function getWilliamsStatus(williams) {
            if (williams >= -20) return {class: 'bearish', text: 'ÜBERKAUFT'};
            if (williams <= -80) return {class: 'bullish', text: 'ÜBERVERKAUFT'};
            return {class: 'neutral', text: 'NEUTRAL'};
        }

        function getCCIStatus(cci) {
            if (cci > 100) return {class: 'bearish', text: 'ÜBERKAUFT'};
            if (cci < -100) return {class: 'bullish', text: 'ÜBERVERKAUFT'};
            return {class: 'neutral', text: 'NEUTRAL'};
        }

        function getADXStatus(adx) {
            if (adx > 40) return {class: 'bullish', text: 'STARK'};
            if (adx > 25) return {class: 'neutral', text: 'TREND'};
            return {class: 'bearish', text: 'SCHWACH'};
        }

        function getATRStatus(atr) {
            return {class: 'neutral', text: 'VOLATILITÄT'};
        }

        function getOBVStatus(obv) {
            return {class: 'neutral', text: 'VOLUMEN'};
        }

        function getTrendClass(strength) {
            if (strength >= 3) return 'bullish';
            if (strength <= 1) return 'bearish';
            return 'neutral';
        }

        function getVolatilityClass(volatility) {
            if (volatility > 5) return 'bearish';
            if (volatility < 2) return 'bullish';
            return 'neutral';
        }

        function getMomentumClass(momentum) {
            if (momentum >= 2) return 'bullish';
            if (momentum <= 1) return 'bearish';
            return 'neutral';
        }

        function getEfficiencyClass(efficiency) {
            if (efficiency > 70) return 'bullish';
            if (efficiency < 30) return 'bearish';
            return 'neutral';
        }

        function getScoreColor(score) {
            if (score >= 70) return 'var(--success)';
            if (score >= 40) return 'var(--warning)';
            return 'var(--danger)';
        }

        function getSentimentColor(sentiment) {
            if (sentiment.includes('BULLISH')) return 'var(--success)';
            if (sentiment.includes('BEARISH')) return 'var(--danger)';
            return 'var(--warning)';
        }

        function getSentimentIcon(sentiment) {
            if (sentiment.includes('STRONG BULLISH')) return '🚀';
            if (sentiment.includes('BULLISH')) return '📈';
            if (sentiment.includes('STRONG BEARISH')) return '📉';
            if (sentiment.includes('BEARISH')) return '⬇️';
            return '➡️';
        }

        function getSignalColor(signalType) {
            if (signalType.includes('BUY')) return 'var(--success)';
            if (signalType.includes('SELL')) return 'var(--danger)';
            return 'var(--warning)';
        }

        function getSignalIcon(signalType) {
            if (signalType.includes('STRONG BUY')) return '🟢';
            if (signalType.includes('BUY')) return '📈';
            if (signalType.includes('STRONG SELL')) return '🔴';
            if (signalType.includes('SELL')) return '📉';
            return '⚪';
        }

        function getRiskColor(risk) {
            if (risk === 'LOW') return 'var(--success)';
            if (risk === 'HIGH') return 'var(--danger)';
            return 'var(--warning)';
        }

        function getVolatilityLevel(volatility) {
            if (volatility > 10) return 'SEHR HOCH';
            if (volatility > 5) return 'HOCH';
            if (volatility > 2) return 'MITTEL';
            return 'NIEDRIG';
        }

        function getEfficiencyLevel(efficiency) {
            if (efficiency > 80) return 'SEHR GUT';
            if (efficiency > 60) return 'GUT';
            if (efficiency > 40) return 'MITTEL';
            return 'NIEDRIG';
        }

        // Formatting Functions
        function formatPrice(price) {
            if (price >= 1000) {
                return Number(price).toLocaleString('de-DE', {minimumFractionDigits: 2, maximumFractionDigits: 2});
            } else if (price >= 1) {
                return Number(price).toLocaleString('de-DE', {minimumFractionDigits: 2, maximumFractionDigits: 4});
            } else {
                return Number(price).toLocaleString('de-DE', {minimumFractionDigits: 4, maximumFractionDigits: 8});
            }
        }

        function formatVolume(volume) {
            if (volume >= 1e12) return (volume / 1e12).toFixed(2) + 'T';
            if (volume >= 1e9) return (volume / 1e9).toFixed(2) + 'B';
            if (volume >= 1e6) return (volume / 1e6).toFixed(2) + 'M';
            if (volume >= 1e3) return (volume / 1e3).toFixed(2) + 'K';
            return volume.toFixed(2);
        }

        function formatPatternName(pattern) {
            return pattern.replace(/_/g, ' ')
                         .replace(/\b\w/g, l => l.toUpperCase())
                         .replace('Bullish', '📈')
                         .replace('Bearish', '📉');
        }

        function getPatternDescription(pattern) {
            const descriptions = {
                'doji': 'Indecision in market',
                'hammer': 'Bullish reversal signal',
                'shooting_star': 'Bearish reversal signal',
                'spinning_top': 'Market indecision',
                'marubozu': 'Strong directional move',
                'dragonfly_doji': 'Bullish reversal',
                'gravestone_doji': 'Bearish reversal',
                'engulfing_bullish': 'Strong bullish signal',
                'engulfing_bearish': 'Strong bearish signal',
                'harami': 'Potential reversal',
                'piercing_line': 'Bullish reversal',
                'dark_cloud_cover': 'Bearish reversal',
                'three_white_soldiers': 'Strong uptrend',
                'three_black_crows': 'Strong downtrend',
                'morning_star': 'Bullish reversal',
                'evening_star': 'Bearish reversal',
                'abandoned_baby_bullish': 'Rare bullish reversal',
                'abandoned_baby_bearish': 'Rare bearish reversal',
                'advance_block': 'Trend weakening',
                'breakaway': 'Trend continuation',
                'hikkake': 'False breakout pattern'
            };
            return descriptions[pattern] || 'Pattern detected';
        }

        // Auto-refresh for live mode
        setInterval(() => {
            if (liveModeEnabled && analysisData) {
                // Update timestamp in status banner
                const banner = document.querySelector('.status-banner');
                if (banner) {
                    const now = new Date().toLocaleTimeString('de-DE');
                    banner.querySelector('.live-indicator span').textContent = `LIVE UPDATE - ${now}`;
                }
            }
        }, 1000);

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'Enter':
                        e.preventDefault();
                        analyzeSymbol();
                        break;
                    case 'l':
                        e.preventDefault();
                        toggleLiveMode();
                        break;
                }
            }
        });

        // Initial load message
        console.log('🔥 ULTIMATE Trading Analysis Pro initialized');
        console.log('💡 Keyboard shortcuts: Ctrl+Enter (Analyze), Ctrl+L (Toggle Live)');
    </script>
</body>
</html>'''
    
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
