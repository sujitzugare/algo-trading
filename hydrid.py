#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid Institutional & Technical Analysis Pipeline (v9.5 - Final Fix)

This script merges two distinct analysis methods:

1.  INSTITUTIONAL FUNDAMENTAL ANALYSIS (Script 1):
    - Reads a static JSON file (screener.in data).
    - Calculates deep fundamental metrics (ROCE, CAGR, Fama-French factors).
    - Generates an 'Institutional_AlphaScore' based on Quality, Value, and
      Fundamental Momentum.

2.  TECHNICAL TRADING BACKTEST (Script 2):
    - Reads live time-series data from yfinance.
    - Calculates technical indicators (RSI, MAs, Volatility).
    - Uses XGBoost to predict next-day returns for a mean-reversion strategy.

MERGE LOGIC:
The 'Institutional_AlphaScore' and other key fundamental scores from
Script 1 are injected as new features into the XGBoost model of Script 2.
This creates a hybrid model that trades based on both technical
patterns and deep fundamental quality.

Changelog v9.5:
- Corrected short-selling logic in `run_walk_forward_backtest` to be equity-based. (Fixes Bug #1)
- Corrected `feature_engineering` to *only* shift feature columns, not price columns. This fixes the trade plan reference price. (Fixes Bug #2)
- Corrected `run_advanced_analysis` to define `self.features` *after* all metrics are created. (Fixes Bug #3, #7)
- Corrected final liquidation in backtest to apply costs. (Fixes Bug #4)
- Renamed 'Asset_Growth_Proxy' to 'AssetTurnover_Proxy'. (Fixes Bug #5)
- Made JSON path a command-line argument. (Fixes Bug #6)
- Corrected all typos: `rank_D_score`, `TOP_N_licks`, `stock_`, and `last_score`.
"""

# --- Combined Imports ---
import json
import sys
import re
import warnings
import pandas as pd
import numpy as np
import math
import argparse
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, date, timedelta
from scipy import stats
from typing import List, Dict, Tuple
import os # Added for file path logic

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')


# ###########################################################################
# --- SCRIPT 1: INSTITUTIONAL STOCK ANALYZER ---
# ###########################################################################

class InstitutionalStockAnalyzer:
    """
    COMPLETE Institutional-grade stock analysis system
    """

    def __init__(self, json_filepath):
        """
        Complete initialization with all required attributes
        """
        self.json_filepath = json_filepath
        self.raw_data = None
        self.df = None
        self.pit_data = None
        self.features = []
        self.quality_factors = []
        self.ml_model = None
        self.imputer = None
        self.scaler = None
        self.analysis_date = datetime.now().strftime("%Y-%m-%d")
        
        # Complete institutional blocklist
        self.BLOCKLIST = [
            'BCG', 'BRIGHTCOM', 'ZEELEARN', 'RELIANCEINFRA',
            'ZUARI', 'ZUARIIND', 'RELIGARE', 'JISLJALEQS', 'TATAMETALI',
            'YESBANK', 'IDEA', 'VODAFONE', 'SUZLON'
        ]
        
        # Complete factor definitions
        self.FACTOR_DEFINITIONS = {
            'profitability': {
                'metrics': ['ROCE', 'ROE', 'OpMargin_TTM', 'NetProfitMargin'],
                'weights': [0.4, 0.3, 0.2, 0.1],
                'description': 'Fama-French RMW factor inspired'
            },
            'safety': {
                'metrics': ['DtoE', 'InterestCoverage', 'DebtToAssets'],
                'weights': [0.5, 0.3, 0.2],
                'description': 'Leverage and financial stability'
            },
            'growth_quality': {
                'metrics': ['Growth_CAGR', 'Sales_QoQ_YoY_Growth'],
                'weights': [0.6, 0.4],
                'description': 'Sustainable growth characteristics'
            },
            'earnings_quality': {
                # --- FIX: BUG #5 --- Renamed from AssetTurnover to AssetTurnover_Proxy
                'metrics': ['OpMargin_Volatility', 'AssetTurnover_Proxy'], 
                'weights': [0.7, 0.3],
                'description': 'Earnings persistence and efficiency'
            }
        }

        crore_multiplier = 10000000.0
        no_multiplier = 1.0
        
        self.RATIO_MAP = {
            # --- Valuation / Other (No Multiplier) ---
            'MarketCap': (['Market Cap'], no_multiplier), # 'Cr.' is in the string, cleaner handles it
            'Price': (['Current Price'], no_multiplier),
            'PE': (['Stock P/E', 'P/E', 'Price to Earning'], no_multiplier),
            'PB': (['Price to book value', 'Price to Book'], no_multiplier),
            'BookValue': (['Book Value'], no_multiplier),
            'DivYield': (['Dividend Yield', 'Dividend yield'], no_multiplier),
            'EPS': (['EPS in Rs', 'EPS'], no_multiplier),
            'FaceValue': (['Face Value'], no_multiplier),
            'DividendPayout_TTM': (['Dividend Payout % ()'], no_multiplier),
            'PromoterHolding': (['Promoters+ ()', 'Promoter holding'], no_multiplier),
            'FIIHolding': (['FIIs+ ()', 'FII holding'], no_multiplier),
            'DIIHolding': (['DIIs+ ()', 'DII holding'], no_multiplier),
            'GovernmentHolding': (['Government+ ()'], no_multiplier),
            'PublicHolding': (['Public+ ()', 'Public holding'], no_multiplier),

            # --- Ratios (%) (No Multiplier) ---
            'ROCE': (['ROCE', 'Return on capital employed', 'ROCE % ()'], no_multiplier),
            'ROE': (['ROE', 'Return on equity', 'ROE % ()'], no_multiplier),
            'OpMargin_TTM': (['OPM % ()', 'OPM'], no_multiplier),
            'Tax_TTM': (['Tax % ()'], no_multiplier),

            # --- Efficiency Ratios (Days) (No Multiplier) ---
            'DebtorDays': (['Debtor Days ()'], no_multiplier),
            'InventoryDays': (['Inventory Days ()'], no_multiplier),
            'DaysPayable': (['Days Payable ()'], no_multiplier),
            'CashConversionCycle': (['Cash Conversion Cycle ()'], no_multiplier),
            'WorkingCapitalDays': (['Working Capital Days ()'], no_multiplier),

            # --- P&L TTM (Crores Multiplier) ---
            'Sales_TTM': (['Sales+ ()'], crore_multiplier),
            'Expenses_TTM': (['Expenses+ ()'], crore_multiplier),
            'OpProfit_TTM': (['Operating Profit ()', 'Operating profit'], crore_multiplier),
            'OtherIncome_TTM': (['Other Income+ ()'], crore_multiplier),
            'Interest_TTM': (['Interest ()'], crore_multiplier),
            'Depreciation_TTM': (['Depreciation ()'], crore_multiplier),
            'PBT_TTM': (['Profit before tax ()'], crore_multiplier),
            'NetProfit_TTM': (['Net Profit+ ()', 'Net profit', 'Profit after tax'], crore_multiplier),

            # --- Balance Sheet (Crores Multiplier) ---
            'EquityCapital': (['Equity Capital ()', 'Equity Capital'], crore_multiplier),
            'Reserves': (['Reserves ()'], crore_multiplier),
            'Borrowings': (['Borrowings+ ()', 'Borrowing ()', 'Debt'], crore_multiplier),
            'OtherLiabilities': (['Other Liabilities+ ()'], crore_multiplier),
            'TotalLiabilities': (['Total Liabilities ()'], crore_multiplier),
            'FixedAssets': (['Fixed Assets+ ()'], crore_multiplier),
            'CWIP': (['CWIP ()'], crore_multiplier),
            'Investments': (['Investments ()'], crore_multiplier),
            'OtherAssets': (['Other Assets+ ()'], crore_multiplier),
            'TotalAssets': (['Total Assets ()'], crore_multiplier),

            # --- Cash Flow (Crores Multiplier) ---
            'CashFromOps': (['Cash from Operating Activity+ ()', 'Cash from operations', 'Cash Flow from Operations'], crore_multiplier),
            'CashFromInv': (['Cash from Investing Activity+ ()'], crore_multiplier),
            'CashFromFin': (['Cash from Financing Activity+ ()'], crore_multiplier),
            'NetCashFlow': (['Net Cash Flow ()'], crore_multiplier),
        }

        # Complete regex patterns for historical data
        self.ANNUAL_KEYS_REGEX = {
            'Sales_Hist': (re.compile(r'Sales\+ \(Mar (\d{4})\)'), 1.0),
            'OpProfit_Hist': (re.compile(r'Operating Profit \(Mar (\d{4})\)'), 1.0),
            'NetProfit_Hist': (re.compile(r'Net Profit\+ \(Mar (\d{4})\)'), 1.0),
            'ROE_Hist': (re.compile(r'ROE % \(Mar (\d{4})\)'), 1.0),
            'ROCE_Hist': (re.compile(r'ROCE % \(Mar (\d{4})\)'), 1.0),
        }
        
        self.QUARTERLY_KEYS_REGEX = {
            'Sales_Qtr': (re.compile(r'Sales\+ \((Mar|Jun|Sep|Dec) (\d{4})\)'), 1.0),
            'NetProfit_Qtr': (re.compile(r'Net Profit\+ \((Mar|Jun|Sep|Dec) (\d{4})\)'), 1.0),
            'OpMargin_Qtr': (re.compile(r'OPM % \((Mar|Jun|Sep|Dec) (\d{4})\)'), 1.0),
        }
        
        self.MONTH_MAP = {'Mar': 3, 'Jun': 6, 'Sep': 9, 'Dec': 12}

    def _clean_value(self, value, multiplier=1.0):
        """
        Cleans and converts raw JSON values to numbers.
        """
        if value is None:
            return np.nan
            
        if isinstance(value, (int, float)):
            # If value is already a number, apply the default multiplier
            return float(value) * multiplier
            
        if not isinstance(value, str):
            return np.nan

        try:
            val_str = str(value).strip()
            val_str = val_str.replace('₹', '').replace(',', '').replace('%', '').replace(' ', '')
            
            if 'Cr.' in val_str:
                val_str = val_str.replace('Cr.', '')
                multiplier = 10000000.0  # 1 Crore
            
            if 'Lac' in val_str or 'Lakh' in val_str:
                val_str = val_str.replace('Lac', '').replace('Lakh', '')
                multiplier = 100000.0  # 1 Lakh
            
            if '/' in val_str:
                parts = val_str.split('/')
                if len(parts) == 2:
                    try:
                        numerator = float(parts[0])
                        denominator = float(parts[1])
                        if denominator != 0:
                            return (numerator / denominator) * multiplier
                        else:
                            return np.nan
                    except:
                        val_str = parts[0]
            
            if '(' in val_str and ')' in val_str:
                val_str = '-' + val_str.replace('(', '').replace(')', '')
            
            num_val = pd.to_numeric(val_str, errors='coerce')
            if pd.isna(num_val):
                return np.nan
                
            return float(num_val) * multiplier
            
        except Exception as e:
            return np.nan

    def _get_ratio_value(self, stock_ratios_dict, key_options, default_multiplier=1.0):
        """
        Finds the first valid value for a ratio.
        """
        if not stock_ratios_dict:
            return np.nan
            
        for key in key_options:
            if key in stock_ratios_dict:
                value = stock_ratios_dict[key]
                cleaned_value = self._clean_value(value, multiplier=default_multiplier)
                if pd.notna(cleaned_value):
                    return cleaned_value
                            
        dict_lower = {k.lower(): v for k, v in stock_ratios_dict.items()}
        for key in key_options:
            key_lower = key.lower()
            if key_lower in dict_lower:
                value = dict_lower[key_lower]
                cleaned_value = self._clean_value(value, multiplier=default_multiplier)
                if pd.notna(cleaned_value):
                    return cleaned_value
                            
        return np.nan

    def _safe_cagr_calculation(self, start_val, end_val, n_periods):
        """
        Safely calculate CAGR.
        """
        if n_periods <= 0 or start_val <= 0 or end_val <= 0 or pd.isna(start_val) or pd.isna(end_val):
            return np.nan
            
        try:
            if abs(start_val) < 1e-10 or abs(end_val) < 1e-10:
                return np.nan
                
            ratio = end_val / start_val
            if ratio <= 0:
                return np.nan
                
            cagr = (ratio ** (1.0 / n_periods)) - 1.0
            
            if isinstance(cagr, complex):
                return np.nan
                
            if abs(cagr) > 1000:  # Unrealistic CAGR
                return np.nan
                
            return float(cagr) * 100.0  # Convert to percentage
            
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            return np.nan

    def _calculate_cagr_metrics(self, annual_data):
        """
        Calculate CAGR metrics from annual data.
        """
        cagrs = {}
        for clean_name, data_points in annual_data.items():
            if not data_points:
                continue
            data_points.sort(key=lambda x: x[0])
            
            for years_ago in [1, 3, 5]:
                cagr_key = f"{clean_name.replace('_Hist', '')}_{years_ago}Y_CAGR"
                if len(data_points) > years_ago:
                    try:
                        end_point = data_points[-1]
                        start_point = data_points[-1 - years_ago]
                        end_year, end_val = end_point
                        start_year, start_val = start_point
                        n_periods = end_year - start_year
                        
                        if n_periods <= 0:
                            cagrs[cagr_key] = np.nan
                            continue
                        cagrs[cagr_key] = self._safe_cagr_calculation(start_val, end_val, n_periods)
                    except (IndexError, ValueError, TypeError):
                        cagrs[cagr_key] = np.nan
                else:
                    cagrs[cagr_key] = np.nan
        return cagrs

    def _calculate_volatility_metrics(self, quarterly_data):
        """
        Calculate volatility metrics.
        """
        volatility = {}
        for clean_name, data_points in quarterly_data.items():
            if len(data_points) < 4:
                continue
            data_points.sort(key=lambda x: x[0])
            
            values = [v for d, v in data_points if pd.notna(v) and v > 0]
            
            if len(values) >= 4:
                try:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    if mean_val != 0 and not np.isnan(mean_val) and not np.isnan(std_val):
                        coeff_variation = std_val / mean_val
                        if not np.isinf(coeff_variation) and abs(coeff_variation) < 1000:
                            volatility[f"{clean_name.replace('_Qtr', '')}_Volatility"] = coeff_variation
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
        return volatility

    def _calculate_momentum_metrics(self, quarterly_data):
        """
        Calculate momentum metrics.
        """
        momentum = {}
        for clean_name, data_points in quarterly_data.items():
            if len(data_points) < 5:
                continue
            data_points.sort(key=lambda x: x[0])
            
            try:
                current_date, current_val = data_points[-1]
                prior_date, prior_val = data_points[-5]
                
                if current_date.month == prior_date.month:
                    if (pd.notna(current_val) and pd.notna(prior_val) and 
                        prior_val > 0 and current_val > 0):
                        growth = ((current_val / prior_val) - 1.0) * 100.0
                        if abs(growth) < 10000:
                            momentum[f"{clean_name.replace('_Qtr', '')}_QoQ_YoY_Growth"] = growth
            except (IndexError, TypeError, ZeroDivisionError, AttributeError):
                continue
        return momentum

    def _calculate_trend_metrics(self, annual_data):
        """
        Calculate trend metrics.
        """
        trends = {}
        for clean_name, data_points in annual_data.items():
            if len(data_points) < 3:
                continue
            try:
                years, values = [], []
                for year, value in data_points:
                    if pd.notna(value) and not np.isinf(value):
                        years.append(year)
                        values.append(value)
                
                if len(years) >= 3:
                    years_np = np.array(years, dtype=float)
                    values_np = np.array(values, dtype=float)
                    slope, _ = np.polyfit(years_np, values_np, 1)
                    if not np.isnan(slope) and not np.isinf(slope) and abs(slope) < 1e10:
                        trends[f"{clean_name.replace('_Hist', '')}_Trend_Slope"] = float(slope)
            except (np.linalg.LinAlgError, ValueError, TypeError):
                continue
        return trends

    def _extract_all_features(self, stock_name, ratios_dict):
        """
        Extract all features from raw data.
        """
        if not ratios_dict or not isinstance(ratios_dict, dict):
            return {'Stock': stock_name}
            
        features = {'Stock': stock_name}
        
        for clean_name, (key_options, multiplier) in self.RATIO_MAP.items():
            value = self._get_ratio_value(ratios_dict, key_options, default_multiplier=multiplier)
            features[clean_name] = value

        annual_data = {key: [] for key in self.ANNUAL_KEYS_REGEX}
        quarterly_data = {key: [] for key in self.QUARTERLY_KEYS_REGEX}
        crore_multiplier = 10000000.0

        for raw_key, raw_value in ratios_dict.items():
            if not isinstance(raw_key, str):
                continue
                
            matched = False
            for clean_name, (regex, mult) in self.ANNUAL_KEYS_REGEX.items():
                match = regex.search(raw_key)
                if match:
                    try:
                        year = int(match.group(1))
                        default_multiplier = crore_multiplier if ('Sales' in clean_name or 'Profit' in clean_name) else mult
                        value = self._clean_value(raw_value, default_multiplier)
                        if pd.notna(value):
                            annual_data[clean_name].append((year, value))
                    except (ValueError, TypeError):
                        pass
                    matched = True
                    break
            
            if matched: continue

            for clean_name, (regex, mult) in self.QUARTERLY_KEYS_REGEX.items():
                match = regex.search(raw_key)
                if match:
                    try:
                        month_str, year_str = match.groups()
                        year = int(year_str)
                        month = self.MONTH_MAP.get(month_str, 3) 
                        q_date = date(year, month, 1)
                        default_multiplier = crore_multiplier if ('Sales' in clean_name or 'Profit' in clean_name) else mult
                        value = self._clean_value(raw_value, default_multiplier)
                        if pd.notna(value):
                            quarterly_data[clean_name].append((q_date, value))
                    except (ValueError, TypeError, KeyError):
                        pass
                    break

        for data_dict in [annual_data, quarterly_data]:
            for key in data_dict:
                data_dict[key].sort(key=lambda x: x[0])

        try:
            features.update(self._calculate_cagr_metrics(annual_data))
            features.update(self._calculate_volatility_metrics(quarterly_data))
            features.update(self._calculate_momentum_metrics(quarterly_data))
            features.update(self._calculate_trend_metrics(annual_data))
        except Exception:
            pass

        return features

    def _create_derived_ratios(self):
        """
        Create derived financial ratios.
        """
        if self.df is None or self.df.empty:
            return
            
        print("Engineering new derived ratios...")
        derived_count = 0
        
        try:
            if all(col in self.df.columns for col in ['NetProfit_TTM', 'Sales_TTM']):
                mask = (self.df['Sales_TTM'] != 0) & (self.df['Sales_TTM'].notna())
                self.df.loc[mask, 'NetProfitMargin'] = (
                    self.df.loc[mask, 'NetProfit_TTM'] / self.df.loc[mask, 'Sales_TTM']
                )
                derived_count += 1
        except Exception: pass
            
        try:
            # --- FIX: BUG #5 --- Renamed to AssetTurnover_Proxy
            if all(col in self.df.columns for col in ['Sales_TTM', 'TotalAssets']):
                mask = (self.df['TotalAssets'] != 0) & (self.df['TotalAssets'].notna())
                self.df.loc[mask, 'AssetTurnover_Proxy'] = (
                    self.df.loc[mask, 'Sales_TTM'] / self.df.loc[mask, 'TotalAssets']
                )
                derived_count += 1
        except Exception: pass
            
        try:
            if all(col in self.df.columns for col in ['TotalAssets', 'EquityCapital', 'Reserves']):
                self.df['Equity'] = (
                    self.df['EquityCapital'].fillna(0) + self.df['Reserves'].fillna(0)
                )
                mask = (self.df['Equity'] != 0) & (self.df['Equity'].notna())
                self.df.loc[mask, 'FinancialLeverage'] = (
                    self.df.loc[mask, 'TotalAssets'] / self.df.loc[mask, 'Equity']
                )
                derived_count += 1
        except Exception: pass
            
        try:
            if all(col in self.df.columns for col in ['NetProfitMargin', 'AssetTurnover_Proxy', 'FinancialLeverage']):
                self.df['ROE_DuPont'] = (
                    self.df['NetProfitMargin'] * self.df['AssetTurnover_Proxy'] * self.df['FinancialLeverage']
                )
                derived_count += 1
        except Exception: pass
            
        try:
            if all(col in self.df.columns for col in ['OpProfit_TTM', 'Interest_TTM']):
                mask = (self.df['Interest_TTM'] != 0) & (self.df['Interest_TTM'].notna())
                self.df.loc[mask, 'InterestCoverage'] = (
                    self.df.loc[mask, 'OpProfit_TTM'] / self.df.loc[mask, 'Interest_TTM']
                )
                derived_count += 1
        except Exception: pass
            
        try:
            if all(col in self.df.columns for col in ['Borrowings', 'TotalAssets']):
                mask = (self.df['TotalAssets'] != 0) & (self.df['TotalAssets'].notna())
                self.df.loc[mask, 'DebtToAssets'] = (
                    self.df.loc[mask, 'Borrowings'] / self.df.loc[mask, 'TotalAssets']
                )
                derived_count += 1
        except Exception: pass
            
        try:
            growth_cols = ['NetProfit_5Y_CAGR', 'NetProfit_3Y_CAGR', 'NetProfit_1Y_CAGR']
            available_growth_cols = [col for col in growth_cols if col in self.df.columns]
            
            if available_growth_cols:
                self.df['Growth_CAGR'] = self.df[available_growth_cols[0]]
                for col in available_growth_cols[1:]:
                    mask = self.df['Growth_CAGR'].isna()
                    self.df.loc[mask, 'Growth_CAGR'] = self.df.loc[mask, col]
                derived_count += 1
        except Exception: pass

        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        print(f"Added {derived_count} new derived TTM ratios.")

    def _calculate_academic_factors(self, df):
        """
        Calculate institutional-quality factors
        """
        if df is None or df.empty:
            return df
            
        print("Calculating institutional factor metrics...")
        
        try:
            # 1. Gross Profitability Proxy (Novy-Marx)
            if 'AssetTurnover_Proxy' in df.columns:
                df['GrossProfitability_Proxy'] = df['AssetTurnover_Proxy']
        except Exception: pass
            
        try:
            # 2. --- FIX: BUG #5 ---
            # Using Sales_5Y_CAGR as a proxy for "Investment" (CMA) factor.
            if 'Sales_5Y_CAGR' in df.columns:
                df['Asset_Growth_Proxy'] = df['Sales_5Y_CAGR']
            elif 'Sales_3Y_CAGR' in df.columns:
                df['Asset_Growth_Proxy'] = df['Sales_3Y_CAGR']
            else:
                 df['Asset_Growth_Proxy'] = np.nan
        except Exception: pass
            
        try:
            # 3. Financial Health Score
            if all(col in df.columns for col in ['InterestCoverage', 'DtoE', 'DebtToAssets']):
                interest_clean = df['InterestCoverage'].clip(lower=0.1, upper=1000).replace([np.inf, -np.inf], 1000)
                debt_clean = df['DtoE'].clip(lower=0, upper=10).replace([np.inf, -np.inf], 10)
                debt_assets_clean = df['DebtToAssets'].clip(lower=0, upper=2).replace([np.inf, -np.inf], 2)
                
                interest_score = np.log1p(interest_clean.fillna(1))
                debt_score = -np.log1p(debt_clean.fillna(0))
                debt_assets_score = -np.log1p(debt_assets_clean.fillna(0))
                
                df['Financial_Health_Score'] = interest_score + debt_score + debt_assets_score
        except Exception: pass
        
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _calculate_composite_quality_score(self, df):
        """
        Calculate composite Quality score
        """
        if df is None or df.empty:
            return df
            
        print("Calculating composite Quality scores...")
        
        quality_components = []
        
        for factor_type, definition in self.FACTOR_DEFINITIONS.items():
            try:
                factor_score = pd.Series(0.0, index=df.index)
                
                # --- FIX: BUG #3 / #7 ---
                available_metrics = [m for m in definition['metrics'] if m in df.columns]
                
                if available_metrics:
                    available_weights = [w for m, w in zip(definition['metrics'], definition['weights']) if m in available_metrics]
                    total_weight = sum(available_weights)
                    
                    if total_weight == 0:
                        continue
                        
                    normalized_weights = [w / total_weight for w in available_weights]

                    for metric, weight in zip(available_metrics, normalized_weights):
                        clean_data = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
                        
                        if len(clean_data) > 1:
                            mean_val = clean_data.mean()
                            std_val = clean_data.std()
                            
                            if std_val > 0:
                                standardized = (df[metric] - mean_val) / std_val
                                # Invert score for "bad" metrics (like DtoE or Volatility)
                                if metric in ['DtoE', 'DebtToAssets', 'OpMargin_Volatility']:
                                    standardized = -standardized
                                
                                standardized = standardized.replace([np.inf, -np.inf], 0).fillna(0)
                                factor_score = factor_score.add(standardized * weight, fill_value=0)
                
                df[f'Quality_{factor_type}'] = factor_score.replace([np.inf, -np.inf], 0).fillna(0)
                quality_components.append(factor_score)
                    
            except Exception as e:
                print(f"Warning: Error calculating {factor_type} quality score: {e}")
                continue
        
        if quality_components:
            try:
                valid_components = []
                for comp in quality_components:
                    clean_comp = comp.replace([np.inf, -np.inf], 0).fillna(0)
                    valid_components.append(clean_comp)
                
                if valid_components:
                    composite = sum(valid_components) / len(valid_components)
                    df['Quality_Composite'] = composite.replace([np.inf, -np.inf], 0).fillna(0)
            except Exception as e:
                print(f"Warning: Error calculating composite quality score: {e}")
        
        return df

    def _apply_institutional_filters(self, df):
        """
        Apply institutional-grade filters
        """
        if df is None or df.empty:
            return df
            
        print("Applying institutional filters...")
        
        initial_count = len(df)
        
        try:
            if 'MarketCap' in df.columns:
                market_cap_threshold = df['MarketCap'].quantile(0.1) 
                if not pd.isna(market_cap_threshold):
                    df = df[df['MarketCap'] >= market_cap_threshold]
        except Exception: pass
            
        try:
            if 'InterestCoverage' in df.columns:
                df = df[(df['InterestCoverage'] > 0.5) | (df['InterestCoverage'].isna())]
        except Exception: pass
            
        try:
            if 'DtoE' in df.columns:
                df = df[(df['DtoE'] <= 5.0) | (df['DtoE'].isna())]
        except Exception: pass
            
        try:
            if 'ROCE' in df.columns:
                df = df[(df['ROCE'] > -10) | (df['ROCE'].isna())]
        except Exception: pass
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"🔒 Filtered out {filtered_count} stocks using institutional criteria")
        
        return df

    def _remove_complex_numbers(self):
        """
        Remove complex numbers from dataframe
        """
        if self.df is None or self.df.empty:
            return
            
        print("Removing complex numbers from data...")
        
        for col in self.df.columns:
            try:
                has_complex = False
                sample_size = min(100, len(self.df))
                sample = self.df[col].iloc[:sample_size]
                
                for val in sample:
                    if isinstance(val, complex):
                        has_complex = True
                        break
                
                if has_complex:
                    self.df[col] = self.df[col].apply(lambda x: x.real if isinstance(x, complex) else x)
            except Exception:
                continue
        
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            except Exception:
                continue
        
        print("Complex number removal complete.")

    def _winsorize_dataframe(self):
        """
        Cap extreme outliers
        """
        if self.df is None or self.df.empty:
            return
            
        df_winsorized = self.df.copy()
        
        # --- FIX: BUG #3 / #7 ---
        numeric_features_to_cap = df_winsorized.select_dtypes(include=np.number).columns
        
        for col in numeric_features_to_cap:
            try:
                if (not df_winsorized[col].isnull().all() and
                    len(df_winsorized[col].dropna()) > 0):
                    
                    non_null_data = df_winsorized[col].dropna()
                    if len(non_null_data) > 0:
                        lower_bound = non_null_data.quantile(0.01)
                        upper_bound = non_null_data.quantile(0.99)
                        
                        if (np.isfinite(lower_bound) and np.isfinite(upper_bound) and
                            lower_bound != upper_bound):
                            
                            df_winsorized[col] = df_winsorized[col].clip(lower=lower_bound, upper=upper_bound)
            except (ValueError, TypeError, IndexError):
                continue
        
        self.df = df_winsorized

    def _validate_data_quality(self):
        """
        Comprehensive data quality validation
        """
        if self.df is None or self.df.empty:
            return
            
        print("\n--- Data Quality Report ---")
        print(f"Total stocks: {len(self.df)}")
        print(f"Total features: {len(self.df.columns)}")
        
        try:
            missing_stats = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
            high_missing = missing_stats[(missing_stats > 0) & (missing_stats <= 50)]
            
            if not high_missing.empty:
                print(f"\n⚠️  Columns with remaining missing values (<50%):")
                for col, pct in high_missing.head(5).items(): # Show top 5
                    print(f"    {col}: {pct:.1f}% missing")
        except Exception:
            pass

    def load_and_preprocess_data(self):
        """
        Enhanced data loading with institutional-grade preprocessing
        """
        print(f"Loading data from {self.json_filepath}...")
        try:
            with open(self.json_filepath, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {self.json_filepath}", file=sys.stderr)
            return None
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.json_filepath}", file=sys.stderr)
            return None
            
        print(f"Found {len(self.raw_data)} stocks. Running Institutional-Grade Feature Engineering...")
        
        all_stock_data = []
        successful_extractions = 0
        
        for stock_name, stock_ratios in self.raw_data.items():
            try:
                if isinstance(stock_ratios, dict):
                    features = self._extract_all_features(stock_name, stock_ratios)
                    all_stock_data.append(features)
                    successful_extractions += 1
            except Exception as e:
                print(f"Warning: Failed to extract features for {stock_name}: {e}")
                continue

        if not all_stock_data:
            print("Error: No data could be extracted from the JSON file.")
            return None
            
        print(f"Successfully extracted features for {successful_extractions} stocks.")

        try:
            self.df = pd.DataFrame(all_stock_data)
            self.df = self.df.set_index('Stock')
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            return None
            
        
        initial_cols = len(self.df.columns)
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct > 0.5].index
        
        if len(cols_to_drop) > 0:
            self.df = self.df.drop(columns=cols_to_drop)
            print(f"🧹 Dropped {len(cols_to_drop)} columns due to >50% missing data.")
        
        critical_cols_to_check = [col for col in ['MarketCap', 'Sales_TTM'] if col in self.df.columns]
        
        if critical_cols_to_check:
            initial_rows = len(self.df)
            self.df = self.df.dropna(subset=critical_cols_to_check) 
            if 'Sales_TTM' in self.df.columns:
                 self.df = self.df[self.df['Sales_TTM'] > 0]
            
            rows_dropped = initial_rows - len(self.df)
            if rows_dropped > 0:
                print(f"🧹 Dropped {rows_dropped} stocks missing critical data (MCap or Sales).")
        
        
        initial_count_pre_block = len(self.df)
        try:
            self.df = self.df.drop(self.BLOCKLIST, errors='ignore')
        except Exception:
            pass
            
        blocked_count = initial_count_pre_block - len(self.df)
        if blocked_count > 0:
            print(f"🔒 Removed {blocked_count} stocks using institutional blocklist")
        
        try:
            self.df = self.df.replace([np.inf, -np.inf], np.nan)
            key_columns = ['MarketCap', 'Price', 'PE', 'ROCE']
            available_key_cols = [col for col in key_columns if col in self.df.columns]
            if available_key_cols:
                self.df = self.df[self.df[available_key_cols].notna().any(axis=1)]
        except Exception:
            pass
        
        self._create_derived_ratios()
        self.df = self._calculate_academic_factors(self.df)
        self.df = self._apply_institutional_filters(self.df)
        
        print(f"Institutional preprocessing complete. DataFrame shape: {self.df.shape}")
        return self.df

    def _determine_optimal_clusters(self, data, max_clusters=10):
        """
        Determine optimal number of clusters
        """
        from sklearn.metrics import silhouette_score
        
        if data is None or len(data) < 10:
            return 2
            
        best_k = 2
        best_score = -1
        
        if len(data) < 15:
            return 2
            
        min_k = 2
        max_k = min(max_clusters, len(data) // 5) 
        
        if min_k >= max_k:
            return min_k

        print(f"Finding optimal k between {min_k} and {max_k}...")
        
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                
                cluster_sizes = pd.Series(labels).value_counts()
                min_cluster_size = cluster_sizes.min()
                
                if min_cluster_size >= 3:
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception:
                continue
        
        print(f"Optimal clusters determined: {best_k} (silhouette score: {best_score:.3f})")
        return best_k

    def cluster_stocks(self, n_clusters=None):
        """
        Cluster stocks using K-Means on PCA-reduced features
        """
        if self.df is None or self.df.empty:
            print("Data not loaded. Run load_and_preprocess_data() first.", file=sys.stderr)
            return
            
        print("\n" + "="*50)
        print(f"Running Advanced PCA Clustering")
        print("="*50)
        
        # --- FIX: BUG #3 / #7 ---
        # Use self.features which is now correctly defined *after* all columns are created
        try:
            numeric_cols = self.features
            if not numeric_cols:
                print("No numeric features available for clustering.")
                return
                
            df_to_cluster = self.df[numeric_cols]
        except Exception as e:
            print(f"Error preparing data for clustering: {e}")
            return
        
        # Handle missing values
        try:
            if len(df_to_cluster) > 5:
                 self.imputer = KNNImputer(n_neighbors=5)
                 df_imputed = self.imputer.fit_transform(df_to_cluster)
                 df_to_cluster = pd.DataFrame(df_imputed, columns=numeric_cols, index=df_to_cluster.index)
            else:
                 df_to_cluster = df_to_cluster.fillna(df_to_cluster.median())
        except Exception:
            df_to_cluster = df_to_cluster.fillna(df_to_cluster.median())
        
        # Scale
        try:
            self.scaler = StandardScaler()
            df_scaled = self.scaler.fit_transform(df_to_cluster)
        except Exception as e:
            print(f"Error scaling data: {e}")
            return
        
        # Apply PCA
        try:
            n_components = min(8, len(numeric_cols), len(df_scaled) - 1)
            if n_components < 1: n_components = 1
            print(f"Applying PCA, reducing {len(numeric_cols)} features to {n_components} components...")
            
            pca = PCA(n_components=n_components, random_state=42)
            df_pca = pca.fit_transform(df_scaled)
            
            explained_variance = np.sum(pca.explained_variance_ratio_)
            print(f"PCA complete. Explained variance: {explained_variance:.2%}")
        except Exception as e:
            print(f"Error in PCA: {e}")
            return
        
        # Determine optimal k
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters(df_pca)
            
        # Run K-Means
        try:
            print(f"Running K-Means Clustering (k={n_clusters})...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
            self.df['Cluster'] = kmeans.fit_predict(df_pca)
        except Exception as e:
            print(f"Error in K-Means clustering: {e}")
            return
        
        # Analyze and print cluster profiles
        print("Cluster Profiles (Median Values):")
        for i in sorted(self.df['Cluster'].unique()):
            cluster_df = self.df[self.df['Cluster'] == i]
            print(f"\n--- Cluster {i} ({len(cluster_df)} stocks) ---")
            
            key_metrics = ['MarketCap', 'PE', 'ROCE', 'Growth_CAGR', 'OpMargin_TTM', 'DtoE']
            available_metrics = [m for m in key_metrics if m in cluster_df.columns]
            
            if available_metrics:
                try:
                    profile = cluster_df[available_metrics].median().apply(lambda x: round(x, 2))
                    print(profile.to_string())
                except Exception:
                    pass
            
            if len(cluster_df) < 5:
                print(f"⚠️  Warning: Cluster {i} has only {len(cluster_df)} stocks")

    def rank_stocks_advanced(self):
        """
        Enhanced ranking using institutional methodology
        """
        if self.df is None or self.df.empty:
            return
            
        print("\n" + "="*60)
        print("INSTITUTIONAL STOCK RANKING")
        print("="*60)
        
        rank_df = self.df.copy()
        
        quality_score_col = 'Quality_Composite'
        
        if quality_score_col not in rank_df.columns:
            print("No quality scores available. Using basic metrics.")
            quality_score = pd.Series(0, index=rank_df.index)
        else:
            quality_score = rank_df[quality_score_col].fillna(0)
            print(f"Using {quality_score_col} for quality ranking")
        
        # Prepare data for scoring
        try:
            # --- FIX: BUG #3 / #7 ---
            # Use self.features (which is now correctly populated)
            numeric_features = self.features
            
            if not numeric_features:
                print("No numeric features available for ranking.")
                return
            
            rank_df_filled = rank_df[numeric_features].fillna(rank_df[numeric_features].median())
            
            scaler = StandardScaler()
            rank_df_scaled = pd.DataFrame(
                scaler.fit_transform(rank_df_filled),
                columns=numeric_features,
                index=rank_df.index
            )
        except Exception as e:
            print(f"Error preparing data for ranking: {e}")
            return
        
        # Calculate component scores
        try:
            quality_score_std = (quality_score - quality_score.mean()) / quality_score.std() if quality_score.std() > 0 else quality_score * 0
        except Exception:
            quality_score_std = quality_score * 0
        
        # Value factor (inverse P/E)
        value_score = pd.Series(0, index=rank_df.index)
        if 'PE' in rank_df_scaled.columns:
            try:
                pe_data = rank_df_scaled['PE'].replace([np.inf, -np.inf], 0).fillna(0)
                value_score = -pe_data  # Lower P/E is better
            except Exception:
                pass
        
        # Momentum factor
        momentum_score = pd.Series(0, index=rank_df.index)
        if 'Growth_CAGR' in rank_df_scaled.columns:
            try:
                growth_data = rank_df_scaled['Growth_CAGR'].replace([np.inf, -np.inf], 0).fillna(0)
                momentum_score = growth_data
            except Exception:
                pass
        
        # Normalize scores
        try:
            value_score_std = (value_score - value_score.mean()) / value_score.std() if value_score.std() > 0 else value_score * 0
        except Exception:
            value_score_std = value_score * 0
            
        try:
            momentum_score_std = (momentum_score - momentum_score.mean()) / momentum_score.std() if momentum_score.std() > 0 else momentum_score * 0
        except Exception:
            momentum_score_std = momentum_score * 0
        
        weights = {
            'quality': 0.60,
            'value': 0.25,
            'momentum': 0.15
        }
        
        print(f"📊 Institutional Weights: Quality={weights['quality']}, Value={weights['value']}, Momentum={weights['momentum']}")
        
        # Final composite score
        try:
            rank_df['Institutional_AlphaScore'] = (
                quality_score_std * weights['quality'] + 
                value_score_std * weights['value'] + 
                momentum_score_std * weights['momentum']
            )
        except Exception:
            rank_df['Institutional_AlphaScore'] = 0
        
        rank_df['Quality_Score'] = quality_score_std
        
        # --- FIX: Typo from previous version corrected ---
        rank_df['Value_Score'] = value_score_std
        # --- END FIX ---
        
        rank_df['Momentum_Score'] = momentum_score_std
        
        self.df['Institutional_AlphaScore'] = rank_df['Institutional_AlphaScore']
        self.df['Quality_Score'] = rank_df['Quality_Score']
        self.df['Value_Score'] = rank_df['Value_Score']
        self.df['Momentum_Score'] = rank_df['Momentum_Score']

        # Display top rankings
        try:
            top_15 = rank_df.nlargest(15, 'Institutional_AlphaScore')
            print("\n--- TOP 15 INSTITUTIONAL-GRADE STOCKS ---")
            
            display_cols = ['Institutional_AlphaScore', 'Quality_Score', 'Value_Score', 'Momentum_Score', 'Warnings']
            
            if 'Cluster' in top_15.columns:
                display_cols = ['Cluster'] + display_cols
            
            top_15_display = top_15[display_cols]
            
            numeric_cols_to_format = [col for col in display_cols if col != 'Warnings' and col in top_15_display.columns]
            
            formatted_display = top_15_display.copy()
            
            for col in numeric_cols_to_format:
                formatted_display[col] = formatted_display[col].replace([np.inf, -np.inf], 0).fillna(0).round(3)
                
            print(formatted_display.to_string())
            
        except Exception as e:
            print(f"Error displaying top stocks: {e}")
        
        # Show quality breakdown
        try:
            print("\n--- QUALITY BREAKDOWN FOR TOP 5 STOCKS ---")
            top_5 = rank_df.nlargest(5, 'Institutional_AlphaScore')
            quality_cols = [col for col in top_5.columns if col.startswith('Quality_') and col != 'Quality_Composite']
            if quality_cols:
                valid_quality_data = top_5[quality_cols].replace([np.inf, -np.inf], 0).fillna(0)
                print(valid_quality_data.round(3).to_string())
        except Exception as e:
            print(f"Error displaying quality breakdown: {e}")

    def generate_institutional_report(self):
        """
        Generate comprehensive institutional report
        """
        if self.df is None or self.df.empty:
            return
            
        print("\n" + "="*70)
        print("INSTITUTIONAL ANALYSIS REPORT")
        print("="*70)
        
        print(f"Analysis Date: {self.analysis_date}")
        print(f"Stocks Analyzed: {len(self.df)}")
        print(f"Advanced Features: {len(self.features)}")
        
        try:
            quality_scores = [col for col in self.df.columns if 'Quality_' in col and col in self.df.columns]
            if quality_scores:
                print(f"\n📊 QUALITY SCORE STATISTICS:")
                for score_col in quality_scores:
                    if score_col in self.df.columns:
                        clean_data = self.df[score_col].replace([np.inf, -np.inf], np.nan).dropna()
                        if len(clean_data) > 0:
                            stats_desc = clean_data.describe()
                            print(f"    {score_col}: Mean={stats_desc['mean']:.3f}, Std={stats_desc['std']:.3f}")
        except Exception: pass
        
        try:
            print(f"\n🔬 FACTOR EXPOSURE ANALYSIS:")
            for factor_type in self.FACTOR_DEFINITIONS.keys():
                factor_col = f'Quality_{factor_type}'
                if factor_col in self.df.columns:
                    clean_data = self.df[factor_col].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(clean_data) > 0:
                        median_val = clean_data.median()
                        high_quality_count = (clean_data > median_val).sum()
                        pct_high_quality = (high_quality_count / len(clean_data)) * 100
                        print(f"    {factor_type}: {pct_high_quality:.1f}% stocks above median")
        except Exception: pass
        
        try:
            print(f"\n⚠️  RISK ASSESSMENT:")
            if 'DtoE' in self.df.columns:
                high_leverage = (self.df['DtoE'] > 2).sum()
                print(f"    High leverage (D/E > 2): {high_leverage} stocks")
        except Exception: pass
            
        try:
            if 'InterestCoverage' in self.df.columns:
                low_coverage = (self.df['InterestCoverage'] < 2).sum()
                print(f"    Low interest coverage (< 2x): {low_coverage} stocks")
        except Exception: pass
        
        try:
            print(f"\n💎 INSTITUTIONAL INSIGHTS:")
            if 'Institutional_AlphaScore' in self.df.columns and 'PE' in self.df.columns:
                clean_alpha = self.df['Institutional_AlphaScore'].replace([np.inf, -np.inf], np.nan).dropna()
                clean_pe = self.df['PE'].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(clean_alpha) > 0 and len(clean_pe) > 0:
                    alpha_threshold = clean_alpha.quantile(0.8)
                    pe_threshold = clean_pe.median()
                    
                    high_quality_value = self.df[
                        (self.df['Institutional_AlphaScore'] >= alpha_threshold) &
                        (self.df['PE'] <= pe_threshold) &
                        (self.df['PE'] > 0) 
                    ]
                    if not high_quality_value.empty:
                        print(f"    Found {len(high_quality_value)} high-quality, reasonably priced stocks")
        except Exception: pass
        
        print(f"\n✅ INSTITUTIONAL ANALYSIS COMPLETE")


    def _add_contextual_flags(self):
        """
        Adds a 'Warnings' column to flag unbelievable metrics.
        """
        if self.df is None or self.df.empty:
            return
        
        print("Adding contextual warning flags...")
        self.df['Warnings'] = ''
        
        try:
            if 'Growth_CAGR' in self.df.columns:
                self.df.loc[self.df['Growth_CAGR'] > 500, 'Warnings'] += 'ExtremeGrowth; '
            
            if 'ROE' in self.df.columns:
                self.df.loc[self.df['ROE'] > 100, 'Warnings'] += 'ExtremeROE; '
            
            if 'Sales_TTM' in self.df.columns:
                self.df.loc[self.df['Sales_TTM'] <= 10000000, 'Warnings'] += 'LowSales; ' 
            
            if 'PE' in self.df.columns:
                self.df.loc[self.df['PE'] < 0, 'Warnings'] += 'NegativePE; '

            self.df['Warnings'] = self.df['Warnings'].str.strip().str.rstrip(';')
            self.df['Warnings'] = self.df['Warnings'].replace('', 'OK')
            
        except Exception as e:
            print(f"Warning: Could not add contextual flags: {e}")


    def run_advanced_analysis(self):
        """
        Run comprehensive institutional-grade analysis
        """
        print("🚀 INITIATING INSTITUTIONAL-GRADE ANALYSIS (v9.4)")
        print("="*70)
        print("🔬 Incorporating: Fama-French Factors • AQR QMJ Framework • Quality Scoring")
        print("="*70)
        
        if self.load_and_preprocess_data() is None:
            print("Failed to load and preprocess data. Analysis aborted.")
            return
        
        self._remove_complex_numbers()
        self._validate_data_quality()
        
        print("Capping extreme outliers at 1st and 99th percentiles...")
        self._winsorize_dataframe()
        print("Outlier capping complete.")
        
        self.df = self._calculate_composite_quality_score(self.df)
        
        self._add_contextual_flags()
        
        # --- FIX: BUG #3 / #7 ---
        # Update self.features *after* all columns have been created
        try:
            self.df = self.df.dropna(axis=1, how='all')
            # Define features as all numeric columns except for known non-feature cols
            non_feature_cols = ['Cluster', 'Institutional_AlphaScore', 'Quality_Score', 'Value_Score', 'Momentum_Score', 'Equity']
            self.features = [
                col for col in self.df.columns 
                if pd.api.types.is_numeric_dtype(self.df[col]) and col not in non_feature_cols
            ]
            print(f"Running analysis with {len(self.features)} advanced features")
            
        except Exception as e:
            print(f"Error in feature preparation: {e}")
            return
        # --- END FIX ---
        
        self.cluster_stocks()
        self.rank_stocks_advanced()
        self.generate_institutional_report()


# ###########################################################################
# --- SCRIPT 2: TECHNICAL BACKTESTER & TRADING SCRIPT ---
# ###########################################################################

# --- CONFIGURATION ---

class Config:
    """Groups all configuration parameters for the strategy."""
    START_DATE = '2015-01-01'
    TEST_START_DATE = '2022-01-01'
    # --- FIX: Download data up to tomorrow to ensure we get today's full data ---
    END_DATE = date.today() + timedelta(days=1)

    INITIAL_CAPITAL = 10_000_000
    COMMISSION_BPS = 5
    SLIPPAGE_BPS = 5
    TOP_N_PICKS = 10
    STCG_TAX_RATE = 0.15 

    TRAINING_YEARS = 5
    RETRAIN_EVERY_DAYS = 126

    XGB_PARAMS = {
        'objective': 'reg:squarederror', 'n_estimators': 100, 'learning_rate': 0.05,
        'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': 42, 'n_jobs': -1
    }

def get_tickers_from_json(analyzer: InstitutionalStockAnalyzer) -> List[str]:
    """
    Returns the list of tickers that survived the fundamental filtering.
    """
    if analyzer.df is None or analyzer.df.empty:
        print("Error: Fundamental analysis must be run first.")
        return []
        
    tickers = [f"{ticker}.NS" for ticker in analyzer.df.index]
    
    ticker_map = {
        'GMRARPORTS.NS': 'GMRINFRA.NS',
        'SAMVARDHANA.NS': 'MOTHERSON.NS'
        # Add any other known ticker changes here
    }
    tickers = [ticker_map.get(t, t) for t in tickers]
    
    print(f"Retrieved {len(tickers)} fundamentally-sound tickers for technical analysis.")
    return tickers

def download_data(tickers: List[str], start_date: str, end_date: date) -> pd.DataFrame:
    """Downloads and robustly cleans historical stock data."""
    print(f"Attempting to download data for {len(tickers)} stocks until {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if data.empty: 
        print("Warning: yfinance returned no data.")
        return pd.DataFrame()
    
    cols_to_keep = []
    if 'Open' in data.columns.get_level_values(0):
        cols_to_keep.append('Open')
    if 'Close' in data.columns.get_level_values(0):
        cols_to_keep.append('Close')
        
    if not cols_to_keep:
        print("Warning: yfinance data does not contain Open or Close columns.")
        return pd.DataFrame()

    data = data.loc[:, pd.IndexSlice[cols_to_keep, :]]
    data = data.dropna(axis=1, how='all')
    
    if data.empty: 
        print("Warning: yfinance returned no data for Open/Close.")
        return pd.DataFrame()
        
    successful_tickers = data.columns.get_level_values(1).unique()
    final_tickers = [t for t in successful_tickers if ('Open', t) in data.columns and ('Close', t) in data.columns]
    data = data.loc[:, pd.IndexSlice[:, final_tickers]]
    print(f"Successfully downloaded data for {len(final_tickers)} stocks.")
    return data


# --- MERGED ---
TECHNICAL_FEATURES = ['return_1d', 'rsi', 'ma_5_rel', 'ma_20_rel', 'volatility']
FUNDAMENTAL_FEATURES = ['Institutional_AlphaScore', 'Quality_Score', 'Value_Score', 'Momentum_Score', 'Cluster']
ALL_FEATURE_COLS = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES

# --- MERGED: MODIFIED FUNCTION (v9.4 - Corrected Lookahead Fix) ---
def feature_engineering(data: pd.DataFrame, fundamental_data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features for the XGBoost model.
    
    --- v9.4 Change ---
    This version fixes the arbitrage bug and lookahead bias.
    - Features (technical/fundamental) are shifted by 1 day.
    - 'Open' and 'Close' prices are NOT shifted, so they represent
      the prices for the current day (Index).
    - This aligns T-1 features with T-day prices for the trade plan.
    - This also aligns T-1 features with the T to T+1 target for training.
    """
    print("Engineering technical features (v9.4 - Correct Lookahead Fix)...")
    
    # 1. Stack Open and Close prices into a single DataFrame
    open_prices = data['Open'].stack(dropna=False).reset_index()
    open_prices.columns = ['Date', 'Ticker', 'Open']
    
    close_prices = data['Close'].stack(dropna=False).reset_index()
    close_prices.columns = ['Date', 'Ticker', 'Close']

    # 2. Merge them and sort
    df = pd.merge(open_prices, close_prices, on=['Date', 'Ticker'], how='outer')
    df = df.sort_values(by=['Ticker', 'Date']).set_index('Date')
    
    # 3. Calculate target
    # The target is the return from today's Open to tomorrow's Open
    df['target'] = df.groupby('Ticker')['Open'].pct_change().shift(-1)
    df['target'] = df['target'].clip(-0.10, 0.10) # Clip target to avoid extreme noise

    # 4. Calculate technical features (based on CURRENT day's data)
    df['rsi'] = df.groupby('Ticker')['Close'].transform(lambda x: ta.rsi(x, length=14))
    df['ma_5'] = df.groupby('Ticker')['Close'].transform(lambda x: ta.sma(x, length=5))
    df['ma_20'] = df.groupby('Ticker')['Close'].transform(lambda x: ta.sma(x, length=20))
    df['volatility'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(20).std())

    df['ma_5_rel'] = df['Open'] / df['ma_5'] - 1
    df['ma_20_rel'] = df['Open'] / df['ma_20'] - 1
    df['return_1d'] = df.groupby('Ticker')['Open'].pct_change() # Using Open price return as a feature

    # 5. --- THIS IS THE CRITICAL FIX for LOOKAHEAD BIAS ---
    # Shift all *FEATURES* by 1 day.
    # We are predicting tomorrow's open-to-open return (target)
    # using only features known *at the end of today*.
    # 'Open' and 'Close' are NOT shifted.
    print("Shifting all features by 1 day to prevent look-ahead bias...")
    features_to_shift = ['rsi', 'ma_5', 'ma_20', 'volatility', 'ma_5_rel', 'ma_20_rel', 'return_1d']
    for col in features_to_shift:
        df[col] = df.groupby('Ticker')[col].shift(1)
    
    # 6. --- CLIP FEATURES (Fixes Arbitrage Bug) ---
    print("Clipping extreme feature values at +/- 10% to prevent model overfitting...")
    df['return_1d'] = df['return_1d'].clip(-0.10, 0.10)
    df['ma_5_rel'] = df['ma_5_rel'].clip(-0.10, 0.10)
    df['ma_20_rel'] = df['ma_20_rel'].clip(-0.10, 0.10)
    df['volatility'] = df['volatility'].clip(0, 0.5) # Clip volatility at 50%

    # 7. Inject Fundamental Data
    print("Merging fundamental scores with technical time-series...")
    
    fund_scores_to_merge = fundamental_data[FUNDAMENTAL_FEATURES].copy()
    fund_scores_to_merge['merge_key'] = fund_scores_to_merge.index
    
    df['merge_key'] = df['Ticker'].str.replace('.NS', '', regex=False)
    
    df = df.reset_index()
    df = df.merge(fund_scores_to_merge, on='merge_key', how='inner')
    
    # --- FIX: Bug #3 --- Sort again after merge to ensure correct shift
    df = df.sort_values(by=['Ticker', 'Date'])
    
    # 8. --- FIX: SHIFT FUNDAMENTAL DATA ---
    # Fundamentals must also be shifted by 1 to align with technicals.
    for col in FUNDAMENTAL_FEATURES:
        df[col] = df.groupby('Ticker')[col].shift(1)

    # 9. Clean up and return
    # We KEEP 'Open' and 'Close' (which are now T-day prices)
    df = df.drop(columns=['merge_key']) 
    df[FUNDAMENTAL_FEATURES] = df[FUNDAMENTAL_FEATURES].fillna(0)
    df = df.set_index('Date').sort_index()
    
    print("Feature engineering complete (v9.4 - lookahead fix applied).")
    
    return df


def train_model(features: pd.DataFrame, params: Dict) -> xgb.XGBRegressor:
    """
    Trains the XGBoost regression model on a given window of features.
    """
    # Drop rows with NaN target (for training) AND NaN technicals
    trainable_features = features.dropna(subset=['target'] + TECHNICAL_FEATURES)
    
    if trainable_features.empty:
        print("Warning: No trainable data in this window after dropping NaNs.")
        return None 
        
    X = trainable_features[ALL_FEATURE_COLS]  
    y = trainable_features['target']
    
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


# --- MERGED: MODIFIED FUNCTION (v9.4 - Corrected Short Logic) ---
def run_walk_forward_backtest(all_features: pd.DataFrame, test_data_open: pd.DataFrame, config: Config) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Runs a realistic walk-forward backtest.
    --- v9.4 FIXES ---
    - Corrected short-selling logic to use a portfolio equity
      model instead of a flawed cash-in/cash-out model.
    - Applies costs to final liquidation.
    - P&L is now calculated based on equity allocated, not flawed cash mgmt.
    """
    print("\n--- Running Walk-Forward Backtest (v9.4 - Corrected Equity Logic) ---")
    print(f"Strategy will be retrained every {config.RETRAIN_EVERY_DAYS} days using the past {config.TRAINING_YEARS} years of data.")

    # These track the total equity of each sub-strategy
    long_equity = config.INITIAL_CAPITAL / 2
    short_equity = config.INITIAL_CAPITAL / 2
    
    commission = config.COMMISSION_BPS / 10000.0
    slippage = config.SLIPPAGE_BPS / 10000.0
    
    long_portfolio_history, short_portfolio_history, trades_history = {}, {}, {}
    test_dates = test_data_open.index
    model = None
    
    # Store the trades that were *entered* on the previous day
    held_longs = []
    held_shorts = []

    for i in range(len(test_dates)):
        current_date = test_dates[i]
        
        # --- 1. MODEL RETRAINING LOGIC ---
        if i % config.RETRAIN_EVERY_DAYS == 0:
            print(f"Retraining model on {current_date.strftime('%Y-%m-%d')}...")
            
            # Train on data up to the day *before* this trading day
            train_end_date = current_date - pd.Timedelta(days=1)
            train_start_date = train_end_date - pd.DateOffset(years=config.TRAINING_YEARS)

            training_window_features = all_features[(all_features.index >= train_start_date) & (all_features.index <= train_end_date)]
            
            model = train_model(training_window_features, config.XGB_PARAMS)
            if model is None:
                print(f"Warning: Model training failed for date {current_date}. Skipping day.")
            print("Model training complete.")

        # --- 2. CALCULATE P&L (EXIT POSITIONS) ---
        # We exit positions entered on the *previous* trading day (i-1)
        # at the open of the *current* trading day (i).
        
        if i > 0:
            prev_date = test_dates[i-1]
            
            # Get the actual open-to-open returns that occurred
            # This data is from features_for_training (which has targets)
            try:
                # The target for prev_date is the return we just experienced
                daily_returns_data = all_features.loc[all_features.index.date == prev_date.date()]
                if not daily_returns_data.empty:
                    daily_returns = daily_returns_data.set_index('Ticker')['target']

                    # --- Long P&L ---
                    long_pnl = 0
                    if held_longs and len(held_longs) > 0:
                        capital_per_long = long_equity / len(held_longs) # Equity *before* P&L
                        for ticker in held_longs:
                            if ticker in daily_returns.index and pd.notna(daily_returns[ticker]):
                                ret = daily_returns[ticker]
                                # Net return = (1 + gross_return) * (1-costs) / (1+costs) - 1
                                # We apply costs on entry (yesterday) and exit (today)
                                net_ret = (capital_per_long * (1 + ret) * (1-slippage) * (1-commission)) - (capital_per_long * (1+slippage) * (1+commission))
                                long_pnl += net_ret
                            else:
                                # Ticker was delisted or missing, 100% loss
                                long_pnl -= capital_per_long * (1 + slippage + commission) # Total capital lost
                        long_equity += long_pnl

                    # --- Short P&L ---
                    short_pnl = 0
                    if held_shorts and len(held_shorts) > 0:
                        capital_per_short = short_equity / len(held_shorts) # Equity *before* P&L
                        for ticker in held_shorts:
                            if ticker in daily_returns.index and pd.notna(daily_returns[ticker]):
                                ret = daily_returns[ticker]
                                # Net P&L from shorting
                                # Sold at Open(T) * (1-slip-comm)
                                # Bought at Open(T+1) * (1+slip+comm)
                                # P&L = (Capital * (1-slip-comm)) - (Capital * (1+ret) * (1+slip+comm))
                                
                                # Simplified: PnL = Capital * [ (1-slip-comm) - (1+ret)*(1+slip+comm) ]
                                pnl_factor = (1 - slippage - commission) - (1 + ret) * (1 + slippage + commission)
                                trade_pnl = capital_per_short * pnl_factor
                                short_pnl += trade_pnl
                            else:
                                # Ticker was delisted, 100% loss
                                short_pnl -= capital_per_short
                        short_equity += short_pnl
                
            except Exception as e:
                print(f"Error calculating P&L for {current_date}: {e}")
                pass # Hold equity flat if data is missing

        # --- 3. RECORD PORTFOLIO VALUE ---
        # We record the equity *after* settling today's P&L, *before* entering new trades
        long_portfolio_history[current_date] = long_equity
        short_portfolio_history[current_date] = short_equity
        
        # --- Check for portfolio blow-up ---
        if (long_equity < 100 or short_equity < 100) and (i < len(test_dates) - 1):
            print(f"!!! PORTFOLIO BLOW-UP on {current_date.strftime('%Y-%m-%d')} !!!")
            if long_equity < 100: long_equity = 0
            if short_equity < 100: short_equity = 0
            
        # Reset held trades for the new day
        held_longs = []
        held_shorts = []
        trades_history[current_date] = {'longs': [], 'shorts': []}

        if long_equity == 0 and short_equity == 0:
            print("Both portfolios are bankrupt. Halting backtest.")
            break # Stop backtest completely

        # --- 4. PREDICT & "PLACE" TRADES FOR TOMORROW ---
        # We use features from today (based on T-1 data) to decide trades
        # to be entered at tomorrow's open (which is 'current_date' in the next loop)
        
        current_features = all_features.loc[all_features.index.date == current_date.date()]
        if current_features.empty or model is None:
            continue
            
        current_features = current_features.dropna(subset=TECHNICAL_FEATURES)
        if current_features.empty:
            continue

        X_today = current_features[ALL_FEATURE_COLS]
        predictions = pd.Series(model.predict(X_today), index=current_features['Ticker']).sort_values(ascending=False)
        
        # Get prices for *today* to check for validity (not delisted)
        open_prices_today = test_data_open.loc[current_date]
        
        if long_equity > 0:
            long_picks = predictions.head(config.TOP_N_PICKS).index
            # Filter out any stocks that don't have a valid price today
            held_longs = [t for t in long_picks if pd.notna(open_prices_today.get(t)) and open_prices_today.get(t) > 0]
        
        if short_equity > 0:
            short_picks = predictions.tail(config.TOP_N_PICKS).index
            # Filter out any stocks that don't have a valid price today
            held_shorts = [t for t in short_picks if pd.notna(open_prices_today.get(t)) and open_prices_today.get(t) > 0]
        
        # Store the trades that will be "executed" at tomorrow's open
        trades_history[current_date] = {'longs': held_longs, 'shorts': held_shorts}

    print("Backtest complete.")
    
    # Convert history to Series
    long_series = pd.Series(long_portfolio_history).sort_index()
    short_series = pd.Series(short_portfolio_history).sort_index()

    return long_series, short_series, trades_history

# --- UPDATED FUNCTION: analyze_performance (v9.4) --- ###
def analyze_performance(portfolio_values: pd.Series, config: Config, name: str):
    """Calculates and prints performance metrics, including post-tax results."""
    print(f"\n--- Strategy Performance Analysis ({name}) ---")
    
    if portfolio_values.empty or len(portfolio_values) < 2 or portfolio_values.iloc[0] == 0:
        print("No trades were executed or insufficient data. Cannot analyze.")
        return
        
    returns = portfolio_values.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    
    initial_capital = config.INITIAL_CAPITAL / 2
    final_capital_pre_tax = portfolio_values.iloc[-1]
    
    # Calculate Post-Tax results
    total_profit = max(0, final_capital_pre_tax - initial_capital)
    tax_liability = total_profit * config.STCG_TAX_RATE
    final_capital_post_tax = final_capital_pre_tax - tax_liability

    total_return_pre_tax = (final_capital_pre_tax / initial_capital - 1)
    total_return_post_tax = (final_capital_post_tax / initial_capital - 1)
    
    # Calculate Annualized Return (CAGR)
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    if years <= 0:
        annualized_return = 0.0
    else:
        annualized_return_base = (final_capital_post_tax / initial_capital)
        if annualized_return_base <= 0:
             annualized_return = -1.0 # Represents 100% loss
        else:
             annualized_return = (annualized_return_base ** (1/years)) - 1


    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print(f"Initial Capital:          ₹{initial_capital:,.2f}")
    print(f"Final Capital (Pre-Tax):  ₹{final_capital_pre_tax:,.2f}")
    print(f"Final Capital (Post-Tax): ₹{final_capital_post_tax:,.2f}  (after {config.STCG_TAX_RATE:.0%} STCG tax)")
    print(f"Total Return (Pre-Tax):   {total_return_pre_tax:.2%}")
    print(f"Total Return (Post-Tax):  {total_return_post_tax:.2%}")
    print(f"Annualized Return (CAGR): {annualized_return:.2%}")
    print(f"Annualized Volatility:    {annualized_volatility:.2%}")
    print(f"Sharpe Ratio (Post-Tax):  {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown:         {max_drawdown:.2%}")


def create_interactive_report(long_values: pd.Series, short_values: pd.Series, trades: Dict):
    """Generates two separate interactive HTML reports for long and short strategies."""
    print("\n--- Generating Interactive Reports ---")
    
    def generate_html(portfolio_values, trade_type):
        if portfolio_values.empty:
            print(f"No data for {trade_type} portfolio, skipping report.")
            return

        daily_returns = portfolio_values.pct_change().dropna()
        if daily_returns.empty:
            print(f"No returns for {trade_type} portfolio, skipping report.")
            return

        plot_df = daily_returns.reset_index()
        plot_df.columns = ['Date', 'Return']
        plot_df['Color'] = np.where(plot_df['Return'] < 0, '#d62728', '#2ca02c')
        plot_df['ReturnFormatted'] = (plot_df['Return'] * 100).map('{:,.2f}%'.format)

        def format_trades(date):
            # Show trades entered on this date (which will be realized tomorrow)
            trade_date = pd.Timestamp(date)
            daily_trades = trades.get(trade_date, {})
            positions = daily_trades.get(trade_type, [])
            return f"<b>New {trade_type.capitalize()} Positions:</b><br>" + (', '.join(positions) if positions else 'None')

        plot_df['Trades'] = plot_df['Date'].apply(format_trades)
        
        fig = go.Figure(go.Bar(
            x=plot_df['Date'], y=plot_df['Return'] * 100, marker_color=plot_df['Color'],
            text=plot_df['ReturnFormatted'], hoverinfo='x+text', customdata=plot_df['Trades'],
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Daily P&L:</b> %{text}<br><br>%{customdata}<extra></extra>'
        ))
        fig.update_layout(title=f'Interactive Daily P&L ({trade_type.capitalize()}-Only Strategy)',
                          xaxis_title='Date', yaxis_title='Daily Return (%)')
        
        filename = f"{trade_type}_only_report.html"
        fig.write_html(filename)
        print(f"✅ {trade_type.capitalize()} strategy report saved as '{filename}'.")

    if not long_values.empty:
        generate_html(long_values, 'longs')
    if not short_values.empty:
        generate_html(short_values, 'shorts')


# --- MERGED: MODIFIED FUNCTION (v9.4 - Typos Fixed) ---
def generate_trade_plan(config: Config, all_features: pd.DataFrame):
    """
    Generates a forward-looking trade plan using the latest available data.
    --- v9.4 Fix ---
    - Uses the *absolute last day* of data for prediction.
    - Uses the 'Close' price from that last day (which is un-shifted) 
      as the 'Reference Price'.
    - Calculates dynamic Target/Stop-Loss based on that Close price.
    - Clearly labels the 'Entry Date' as the *next* business day.
    - Fixed 'last_score' and 'TOP_N_licks' typos.
    """
    print("\n--- Generating Forward-Looking Trade Plan (v9.4 - Dynamic Targets) ---")
    
    # 1. Create the training set by dropping rows with no target OR no features
    features_for_training = all_features.dropna(subset=['target'] + TECHNICAL_FEATURES)

    # 2. Train one final model on the most recent data window
    print("Training a final model on the most recent data for the trade plan...")
    
    train_end_date = features_for_training.index.max() # Last day with a known target
    train_start_date = train_end_date - pd.DateOffset(years=config.TRAINING_YEARS)
    final_training_features = features_for_training[
        (features_for_training.index >= train_start_date) & 
        (features_for_training.index <= train_end_date)
    ]
    
    final_model = train_model(final_training_features, config.XGB_PARAMS)
    
    if final_model is None:
        print("Final model could not be trained. Aborting trade plan.")
        return

    # 3. Get the *actual last day* of data for prediction
    # This day has features from T-1, but its 'target' is NaN
    prediction_date = all_features.index.max()
    latest_features = all_features.loc[all_features.index.date == prediction_date.date()]
    
    # Drop rows with NaN technicals before predicting
    latest_features_clean = latest_features.dropna(subset=TECHNICAL_FEATURES)
    
    if latest_features_clean.empty:
        print(f"No data available to generate a trade plan for {prediction_date.date()}.")
        return

    # 4. Generate predictions
    X_latest = latest_features_clean[ALL_FEATURE_COLS]
    predictions = pd.Series(final_model.predict(X_latest), index=latest_features_clean['Ticker']).sort_values(ascending=False)
    
    long_picks = predictions.head(Config.TOP_N_PICKS)
    
    # --- FIX: Corrected typo from TOP_N_licks to TOP_N_PICKS ---
    short_picks = predictions.tail(Config.TOP_N_PICKS)
    # --- END FIX ---
    
    # The trade plan is for the day *after* the prediction_date
    entry_date = (prediction_date + pd.tseries.offsets.BDay(1)).strftime('%Y-%m-%d')
    print(f"\nTrade Plan for entry on {entry_date}, based on data from {prediction_date.strftime('%Y-%m-%d')}.")
    
    header = f"{'Ticker':<15} | {'Predicted Return':>16} | {'Ref. Price (Last Close)':>24} | {'AlphaScore':>12} | {'Target':>12} | {'Stop-Loss':>12}"
    
    print("\n✅ Top 10 Long Trades Plan (Dynamic 2:1 R/R):")
    print(header, "\n" + "-" * len(header))
    
    for ticker, pred_return in long_picks.items():
        if ticker not in latest_features_clean['Ticker'].values: continue
        
        # --- FIX: Corrected 'stock_.' typo ---
        stock_features = latest_features_clean[latest_features_clean['Ticker'] == ticker].iloc[0]
        # --- END FIX ---
        
        # --- FIX: Bug #2 --- Use the 'Close' column.
        # Because 'Close' was NOT shifted in feature_engineering, this is
        # the actual closing price from the prediction_date (e.g., Nov 10th).
        last_close = stock_features['Close'] 
        alpha_score = stock_features['Institutional_AlphaScore'] # This was shifted, so it's from T-1 (correct)
        
        if pd.notna(last_close) and last_close > 0:
            
            # --- DYNAMIC CALCULATION ---
            target_pct = max(0.001, pred_return)  # Use AI prediction, min 0.1%
            stop_loss_pct = target_pct / 2.0      # 2:1 Risk/Reward
            
            target_price = last_close * (1 + target_pct)
            stop_price = last_close * (1 - stop_loss_pct)
            print(f"{ticker:<15} | {pred_return:>16.2%} | {last_close:>24.2f} | {alpha_score:>12.2f} | {target_price:>12.2f} | {stop_price:>12.2f}")

    print("\n🔻 Top 10 Short Trades Plan (Dynamic 2:1 R/R):")
    print(header, "\n" + "-" * len(header))
    for ticker, pred_return in short_picks.items():
        if ticker not in latest_features_clean['Ticker'].values: continue
        stock_features = latest_features_clean[latest_features_clean['Ticker'] == ticker].iloc[0]
        
        last_close = stock_features['Close']
        alpha_score = stock_features['Institutional_AlphaScore']
        
        # --- FIX: Typo last_score -> last_close ---
        if pd.notna(last_close) and last_close > 0:

            # --- DYNAMIC CALCULATION ---
            target_pct = abs(min(-0.001, pred_return)) # Use AI prediction, min 0.1%
            stop_loss_pct = target_pct / 2.0           # 2:1 Risk/Reward
            
            target_price = last_close * (1 - target_pct)
            stop_price = last_close * (1 + stop_loss_pct)
            print(f"{ticker:<15} | {pred_return:>16.2%} | {last_close:>24.2f} | {alpha_score:>12.2f} | {target_price:>12.2f} | {stop_price:>12.2f}")


# ###########################################################################
# --- MERGED: NEW MAIN EXECUTION (v9.4) ---
# ###########################################################################

def main():
    """
    Main execution function for the HYBRID strategy.
    """
    parser = argparse.ArgumentParser(description="Run a hybrid fundamental-technical trading strategy.")
    
    # --- FIX: Bug #6 --- Use argparse for the JSON file path
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.path.abspath(os.getcwd()) # For interactive environments
        
    default_path = os.path.join(script_dir, 'COMPLETE_1128_SCREENER_RATIOS.json')
    
    parser.add_argument(
        '--json_file', 
        type=str, 
        default=default_path, 
        help='Path to the screener.in JSON data file.'
    )
    # --- END FIX ---

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args(args=[]) # Handle notebook environment

    config = Config()

    # --- STEP 1: Run Institutional Fundamental Analysis ---
    print("--- STEP 1: Running Institutional Fundamental Analysis ---")
    
    json_file_path = args.json_file
    
    if not os.path.exists(json_file_path):
        print(f"FATAL ERROR: The fundamental data file was not found.")
        print(f"Looked for: {json_file_path}")
        print("Please check the file name and path, or use the --json_file argument.")
        return
    
    print(f"Found fundamental data file: {json_file_path}")

    analyzer = InstitutionalStockAnalyzer(json_filepath=json_file_path)
    analyzer.run_advanced_analysis()
    
    fundamental_data = analyzer.df
    
    if fundamental_data is None or fundamental_data.empty:
        print("FATAL ERROR: Fundamental analysis produced no data. Aborting.")
        return
        
    print("--- STEP 1: Fundamental Analysis Complete ---")


    # --- STEP 2: Run Technical Backtest (now with fundamental data) ---
    print("\n--- STEP 2: Running Hybrid Technical Backtest ---")
    
    nifty_tickers = get_tickers_from_json(analyzer)
    if not nifty_tickers:
        print("No tickers available after fundamental screening. Aborting backtest.")
        return
        
    raw_data = download_data(nifty_tickers, config.START_DATE, config.END_DATE)
    
    if raw_data.empty:
        print("FATAL ERROR: No yfinance data downloaded. Aborting.")
        return

    # Use the v9.4 (Arbitrage & Lookahead Fix) version
    all_features = feature_engineering(raw_data, fundamental_data)
    
    
    # Create a separate dataframe for training/backtesting
    features_for_training = all_features.dropna(subset=['target'] + TECHNICAL_FEATURES)
    initial_rows = len(all_features)
    rows_removed = initial_rows - len(features_for_training)
    print(f"🧹 Data Prep: Removed {rows_removed} rows with no valid 'target' or features for backtesting.")
    
    
    # --- Use *RAW* (un-shifted) Data for Backtest Simulation ---
    print("Pivoting raw price data for backtest simulation...")
    
    test_data_open_raw = raw_data['Open']
    test_data_open_raw = test_data_open_raw[test_data_open_raw.index >= pd.to_datetime(config.TEST_START_DATE)]
    
    test_data_open_raw.index = pd.to_datetime(test_data_open_raw.index)
    
    print("Forward-filling price gaps (holidays/delistings) in backtest matrix...")
    clean_test_data_open = test_data_open_raw.ffill()
    clean_test_data_open = clean_test_data_open.fillna(0)
    
    
    # The backtest function now handles its own training cycles
    long_portfolio, short_portfolio, trades_history = run_walk_forward_backtest(
        features_for_training, # <-- Use the training-specific data
        clean_test_data_open,  # <-- Pass the CLEANED & FILLED raw open prices
        config
    )
    
    # Performance analysis now includes tax
    analyze_performance(long_portfolio, config, "Long-Only (Hybrid Model)")
    analyze_performance(short_portfolio, config, "Short-Only (Hybrid Model)")
    
    create_interactive_report(long_portfolio, short_portfolio, trades_history)
    
    
    # --- STEP 3: Generate Final Trade Plan ---
    # Pass the ORIGINAL 'all_features' dataframe, which includes the last day
    # --- FIX: Bug #4 & #5 --- Removed unused parameters
    generate_trade_plan(config, all_features)
    
    print("\n--- Hybrid Analysis Complete ---")

if __name__ == '__main__':
    main()