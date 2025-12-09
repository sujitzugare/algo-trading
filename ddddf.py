import json
import pandas as pd
import numpy as np
import logging
import os
import time
import concurrent.futures
import warnings
from datetime import datetime

# --- Configuration ---
# Suppress warnings and set logging to only show crucial info
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- Library Checks ---
try:
    import yfinance as yf
    # Silence yfinance's own internal error messages
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

class HedgeFundStrategy:
    """Mathematical models for scoring stocks."""
    
    @staticmethod
    def quantitative_momentum(df: pd.DataFrame) -> pd.Series:
        """Scores stocks on price momentum + fundamental growth."""
        score = (
            0.3 * df['ProfitGrowth_CAGR'].rank(pct=True).fillna(0.5) +
            0.3 * df['SalesGrowth_CAGR'].rank(pct=True).fillna(0.5) +
            0.2 * df['ROCE'].rank(pct=True).fillna(0.5) +
            0.2 * df['momentum_12m'].rank(pct=True).fillna(0.5)
        )
        return score
    
    @staticmethod
    def deep_value(df: pd.DataFrame) -> pd.Series:
        """Finds undervalued stocks (Low PE, High Dividend) but avoids junk."""
        # Invert PE rank (Lower PE = Higher Rank)
        pe_rank = df['PE'].rank(pct=True).fillna(1.0)
        
        score = (
            0.4 * (1 - pe_rank) + 
            0.3 * (1 - df['PEG_Ratio'].rank(pct=True).fillna(1.0)) + 
            0.3 * df['DivYield'].rank(pct=True).fillna(0)
        )
        return score

    @staticmethod
    def quality_growth(df: pd.DataFrame) -> pd.Series:
        """Finds compounders: High ROCE, Low Debt, consistent margins."""
        # Invert Debt rank (Lower Debt = Higher Rank)
        debt_rank = df['DebtToEquity'].rank(pct=True).fillna(0.5)
        
        score = (
            0.3 * df['ROCE'].rank(pct=True).fillna(0) +
            0.2 * df['ROE'].rank(pct=True).fillna(0) +
            0.2 * df['OPM'].rank(pct=True).fillna(0) +
            0.2 * (1 - debt_rank) +
            0.1 * df['InterestCoverage'].rank(pct=True).fillna(0.5)
        )
        return score

class AdvancedStockAnalyzer:
    def __init__(self, json_filename: str, cache_days: int = 1):
        self.json_file_path = json_filename
        self.df = pd.DataFrame()
        # Load data (Force refresh)
        self.load_data()
    
    def _clean_value(self, value):
        """Robust cleaner for Indian financial formats."""
        if pd.isna(value) or value in ['', 'N/A', '-', 'nan']: return np.nan
        if isinstance(value, (int, float)): return float(value)
        
        s = str(value).upper().replace(',', '').replace('%', '').replace('₹', '')
        try:
            if 'CR' in s: return float(s.replace('CR', '')) * 10000000
            if 'L' in s: return float(s.replace('L', '')) * 100000
            if 'K' in s: return float(s.replace('K', '')) * 1000
            return float(s)
        except:
            return np.nan

    def fetch_yahoo_parallel(self, tickers: list) -> dict:
        """Fetches live data silently."""
        data_map = {}
        if not YFINANCE_AVAILABLE: return data_map

        print(f"📡 Fetching live data for {len(tickers)} stocks...")
        
        def fetch(t):
            try:
                sym = f"{t}.NS" if not t.endswith('.NS') else t
                ticker = yf.Ticker(sym)
                info = ticker.info
                hist = ticker.history(period="1y")
                
                mom = 0.0
                if not hist.empty:
                    mom = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1

                return t, {
                    'current_price_yahoo': info.get('currentPrice', info.get('previousClose', np.nan)),
                    'mcap': info.get('marketCap', np.nan),
                    'momentum_12m': mom,
                    'beta': info.get('beta', np.nan)
                }
            except:
                return t, {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(fetch, t): t for t in tickers}
            for future in concurrent.futures.as_completed(future_to_url):
                t, res = future.result()
                if res: data_map[t] = res
        return data_map

    def load_data(self):
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {self.json_file_path} not found.")
            return

        yahoo_data = self.fetch_yahoo_parallel(list(raw_data.keys()))

        processed = []
        for ticker, metrics in raw_data.items():
            row = {'Ticker': ticker}
            mappings = {
                'Price': ['Current Price', 'Price'],
                'PE': ['Stock P/E', 'P/E'],
                'ROCE': ['ROCE %', 'ROCE'],
                'ROE': ['Return on equity', 'ROE'],
                'OPM': ['OPM %'],
                'DebtToEquity': ['Debt to Equity', 'D/E'],
                'DivYield': ['Dividend Yield'],
                'Sales_2024': ['Sales+ (Mar 2024)', 'Sales'],
                'Sales_2021': ['Sales+ (Mar 2021)'],
                'Profit_2024': ['Net Profit+ (Mar 2024)', 'Net Profit'],
                'Profit_2021': ['Net Profit+ (Mar 2021)'],
                'InterestCoverage': ['Interest Coverage Ratio']
            }
            
            for field, keys in mappings.items():
                val = np.nan
                for k in keys:
                    if k in metrics:
                        val = self._clean_value(metrics[k])
                        break
                row[field] = val

            if ticker in yahoo_data:
                row.update(yahoo_data[ticker])
                if pd.notna(row.get('current_price_yahoo')):
                    row['Price'] = row['current_price_yahoo']

            processed.append(row)

        self.df = pd.DataFrame(processed)
        self._post_process()

    def _post_process(self):
        """Filters junk and calculates derived metrics."""
        # 1. Convert columns to numeric
        cols = ['Price', 'PE', 'ROCE', 'ROE', 'OPM', 'DebtToEquity', 'DivYield', 'InterestCoverage']
        for c in cols:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

        # 2. FILTER: Remove Penny Stocks & Junk
        if not self.df.empty:
            initial_count = len(self.df)
            self.df = self.df[self.df['Price'] > 20] # Filter < 20
            self.df = self.df[self.df['Price'].notna()]
            print(f"🧹 Filtered {initial_count - len(self.df)} junk/delisted stocks.")

            # 3. Calculate Growth
            self.df['ProfitGrowth_CAGR'] = ((self.df['Profit_2024'] / self.df['Profit_2021'].replace(0, np.nan)) ** (1/3) - 1) * 100
            self.df['SalesGrowth_CAGR'] = ((self.df['Sales_2024'] / self.df['Sales_2021'].replace(0, np.nan)) ** (1/3) - 1) * 100
            
            # 4. Handle NaNs
            self.df['PEG_Ratio'] = self.df['PE'] / self.df['ProfitGrowth_CAGR'].replace(0, 0.1)
            self.df['DivYield'] = self.df.get('DivYield', pd.Series(0, index=self.df.index)).fillna(0)
            
            # --- FIX FOR MOMENTUM ERROR ---
            if 'momentum_12m' not in self.df.columns:
                self.df['momentum_12m'] = 0.0
            self.df['momentum_12m'] = self.df['momentum_12m'].fillna(0)

    def run_strategies(self):
        strategies = {
            '⚡ Quantitative Momentum': HedgeFundStrategy.quantitative_momentum,
            '💎 Quality Growth': HedgeFundStrategy.quality_growth,
            '💰 Deep Value': HedgeFundStrategy.deep_value
        }
        
        all_picks = []
        print("\n" + "="*50)
        for name, func in strategies.items():
            self.df['Score'] = func(self.df)
            top = self.df.nlargest(5, 'Score')
            top['Strategy'] = name
            all_picks.append(top)
            
            print(f"\n{name} (Top 5)")
            print("-" * 50)
            # Safe print
            cols = ['Ticker', 'Price', 'PE', 'ROCE', 'Score']
            valid_cols = [c for c in cols if c in top.columns]
            print(top[valid_cols].to_markdown(index=False, floatfmt=".2f"))

        print("\n" + "="*50)
        print("🏆 FINAL ALADDIN PORTFOLIO (Aggregated)")
        print("-" * 50)
        if all_picks:
            combined = pd.concat(all_picks).drop_duplicates(subset=['Ticker'])
            final = combined.nlargest(10, 'Score')[['Ticker', 'Price', 'Strategy', 'Score']]
            print(final.to_markdown(index=False, floatfmt=".2f"))

# --- Execution ---
if __name__ == "__main__":
    analyzer = AdvancedStockAnalyzer('COMPLETE_1128_SCREENER_RATIOS.json', cache_days=0)
    if not analyzer.df.empty:
        analyzer.run_strategies()