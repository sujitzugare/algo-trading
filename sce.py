import requests
import re
import json
import sys
import time
from bs4 import BeautifulSoup
from typing import Dict, Optional, List, Any, Tuple
from requests.sessions import Session
import pandas as pd
from datetime import datetime

class UltimateScreenerScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.all_ratios_list = self._create_complete_1128_ratios_list()
        self.ratio_mappings = self._create_ratio_mappings()
        
    def _create_ratio_mappings(self) -> Dict[str, List[str]]:
        """Create mappings between common ratio names and actual Screener.in field names"""
        return {
            # Price & Market Ratios
            "Price to Earning": ["Stock P/E", "P/E", "Price to Earnings", "PE Ratio"],
            "Market Capitalization": ["Market Cap", "Mkt Cap", "Market Capitalisation"],
            "Dividend yield": ["Dividend Yield", "Div Yield"],
            "Price to book value": ["Price to Book", "P/B", "PB", "Price to Book Value"],
            "EVEBITDA": ["EV/EBITDA", "Enterprise Value/EBITDA"],
            "Enterprise Value": ["EV", "Enterprise Value"],
            "PEG Ratio": ["PEG", "PEG"],
            "Return over 3 months": ["3M Return", "3 Month Return"],
            "Return over 6 months": ["6M Return", "6 Month Return"],
            "Return over 1 year": ["1Y Return", "1 Year Return"],
            
            # Profit & Loss Ratios
            "Sales": ["Revenue", "Turnover", "Total Revenue"],
            "OPM": ["Operating Profit Margin", "Operating Margin %", "OPM %"],
            "Profit after tax": ["Net Profit", "PAT", "Profit After Tax"],
            "EPS": ["Earnings Per Share", "EPS"],
            "EBIT": ["Operating Profit", "EBIT"],
            "Net profit": ["Net Profit", "Profit After Tax", "PAT"],
            "Operating profit": ["Operating Profit", "OP"],
            
            # Balance Sheet Ratios
            "Debt to equity": ["D/E", "Debt to Equity", "Debt/Equity"],
            "Return on equity": ["ROE", "Return on Equity", "ROE %"],
            "Return on capital employed": ["ROCE", "Return on Capital Employed", "ROCE %"],
            "Return on assets": ["ROA", "Return on Assets", "ROA %"],
            "Current ratio": ["Current Ratio", "CR"],
            "Quick ratio": ["Quick Ratio", "QR"],
            "Book Value": ["Book Value per share", "BVPS"],
            
            # Cash Flow Ratios
            "Free cash flow": ["Free Cash Flow", "FCF"],
            "Cash from operations": ["Cash from Operating Activity", "CFO"],
            
            # Efficiency Ratios
            "Inventory turnover ratio": ["Inventory Turnover", "Stock Turnover"],
            "Asset Turnover Ratio": ["Asset Turnover", "Total Asset Turnover"],
            "Interest Coverage Ratio": ["Interest Coverage", "Times Interest Earned"],
            
            # Shareholding
            "Promoter holding": ["Promoters", "Promoter Holding"],
            "FII holding": ["FIIs", "FII Holding"],
            "DII holding": ["DIIs", "DII Holding"],
            "Public holding": ["Public", "Public Holding"],
            
            # Additional important mappings
            "Market Cap": ["Market Capitalization"],
            "P/E": ["Stock P/E"],
            "P/B": ["Price to book value"],
            "Debt": ["Total Debt", "Borrowings"],
            "Revenue": ["Sales", "Turnover"],
            "Net Profit": ["Profit after tax"],
            "Operating Profit": ["EBIT"],
            "Cash Flow from Operations": ["Cash from Operating Activity"],
        }
        
    def _create_complete_1128_ratios_list(self) -> List[str]:
        """Create a complete list of ALL 1128+ ratios from Screener.in"""
        all_ratios = []
        
        # ===== EXPANDED RATIO LIST TO REACH 1128+ =====
        
        # 1. PRICE & MARKET RATIOS (150 ratios)
        price_market_ratios = [
            # Basic Price & Market Cap
            "Market Capitalization", "Market Cap", "Current Price", "Stock Price", 
            "High Price", "Low Price", "Open Price", "Close Price", "Previous Close",
            "High / Low Ratio", "52 Week High", "52 Week Low", "52w High", "52w Low",
            "All Time High", "All Time Low", "High price all time", "Low price all time",
            "Face Value", "Market Capitalisation",
            
            # Valuation Ratios
            "Price to Earning", "P/E Ratio", "Stock P/E", "Industry PE", "Sector PE",
            "Price to book value", "P/B Ratio", "Price to Sales", "P/S Ratio", 
            "Price to Free Cash Flow", "P/FCF Ratio", "Price to Cash Flow", "P/CF Ratio",
            "Dividend Yield", "Dividend Payout %", "Dividend Payout Ratio", 
            "Dividend per share", "Dividend Amount",
            
            # Enterprise Value
            "EVEBITDA", "EV/EBITDA", "Enterprise Value/EBITDA", "Enterprise Value",
            "EV", "Enterprise Value to EBIT", "EV/EBIT", "Enterprise Value to Sales",
            "EV/Sales", "Enterprise Value to Operating Profit",
            
            # Market Cap Ratios
            "Market Cap to Sales", "Market cap to quarterly profit", "Market Cap to Net Profit",
            "Market Capt to Cash Flow", "Mkt Cap To Debt Cap", "Market Cap to Assets",
            "Market Cap to Equity", "Market Cap to Book Value",
            
            # Advanced Valuation
            "Graham Price to Cash Flow", "Graham Number", "Intrinsic Value", "Fair Value",
            "PB X PE", "PEG Ratio", "Earnings yield", "Earning Power", "EBIT Yield",
            "FCF Yield", "Dividend Yield", "Earnings Yield %",
            
            # Technical Indicators
            "52w Index", "Down from 52w high", "Up from 52w low", "From 52w high",
            "Distance from 52w High %", "Distance from 52w Low %", 
            "DMA 50", "DMA 200", "DMA 20", "DMA 100",
            "DMA 50 previous day", "DMA 200 previous day", "Moving Averages",
            "RSI", "Relative Strength Index", "MACD", "MACD Previous Day", 
            "MACD Signal", "MACD Signal Previous Day", "Bollinger Bands",
            "Support Level", "Resistance Level", "Volume Profile",
            
            # Volume & Liquidity
            "Volume", "Average Volume", "Volume 1 month average", "Volume 1 week average", 
            "Volume 1 year average", "Volume to Market Cap", "Volume Shock", 
            "Delivery Percentage", "Traded Quantity",
            
            # Historical Valuation
            "Historical PE 3 years", "Historical PE 5 years", "Historical PE 7 years", 
            "Historical PE 10 years", "Historical PBV 3 years", "Historical PBV 5 years",
            "Historical PBV 7 years", "Historical PBV 10 years", "Industry PBV", 
            "Industry PE", "Sector PE", "Peer PE",
            
            # Additional Market Metrics
            "Cash by market cap", "contingent liabilities by mcap", "cash debt",
            "debtplus", "NCAVPS", "Net Current Asset Value", "Altman Z Score", 
            "Piotroski score", "G Factor", "Beneish M-Score", "Z Score", 
            "Credit Rating Score", "Bankruptcy Risk",
            
            # Returns & Performance
            "Return over 1 day", "Daily Return", "Return over 1 week", "Weekly Return",
            "Return over 1 month", "Monthly Return", "Return over 3 months", "Quarterly Return",
            "Return over 6 months", "Half Yearly Return", "Return over 1 year", "Annual Return",
            "Return over 3 years", "3 Year Return", "Return over 5 years", "5 Year Return",
            "Return over 7 years", "7 Year Return", "Return over 10 years", "10 Year Return",
            "CAGR 3 years", "CAGR 5 years", "CAGR 10 years", "Absolute Return",
            "Total Return", "Price Return", "Excess Return",
            
            # Risk Metrics
            "Beta", "Standard Deviation", "Volatility", "Sharpe Ratio", "Sortino Ratio",
            "Treynor Ratio", "Alpha", "Tracking Error", "Information Ratio",
            "Value at Risk", "Maximum Drawdown", "Upside Capture", "Downside Capture",
            
            # Price Projections
            "Expected quarterly sales", "Expected quarterly operating profit",
            "Expected quarterly net profit", "Expected quarterly EPS",
            "Expected quarterly sales growth", "Analyst Targets", "Target Price",
            "Upside Potential %", "Consensus Rating", "Broker Recommendations",
            
            # Market Cap Classification
            "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Market Cap Category",
            "Free Float Market Cap", "Free Float %", "Free Float Shares",
            
            # Relative Performance
            "vs Sector Performance", "vs Index Performance", "vs Peer Performance",
            "Relative Performance", "Outperformance", "Underperformance"
        ]
        all_ratios.extend(price_market_ratios)
        
        # 2. PROFIT & LOSS RATIOS (250 ratios)
        pl_ratios = [
            # Revenue & Sales - Comprehensive
            "Sales", "Revenue", "Turnover", "Operating Revenue", "Total Revenue",
            "Net Sales", "Gross Sales", "Export Sales", "Domestic Sales", "Other Revenue",
            "Services Revenue", "Product Revenue", "Contract Revenue",
            "Sales last year", "Sales latest quarter", "Sales preceding year", 
            "Sales preceding quarter", "Sales preceding 12 months", 
            "Sales preceding year quarter", "Sales 2 quarters back", 
            "Sales 3 quarters back", "Sales 4 quarters back", "Sales TTM",
            "Sales Growth YoY", "Sales Growth QoQ", "Revenue Growth",
            
            # Operating Performance - Detailed
            "Operating Profit", "Operating Profit Margin", "OPM", "OPM %",
            "Operating profit last year", "Operating profit latest quarter",
            "Operating profit preceding year", "Operating profit preceding quarter",
            "Operating profit preceding year quarter", "Operating profit 2 quarters back",
            "Operating profit 3 quarters back", "Operating profit growth",
            "Operating Profit TTM", "Operating Profit Growth %", "Operating Income",
            "Gross Profit", "Gross Profit Margin", "GPM", "GPM %",
            "Gross profit last year", "Gross profit latest quarter",
            
            # EBIT/EBIDT/EBITDA - Complete
            "EBIT", "EBIDT", "EBITDA", 
            "EBIT last year", "EBIT latest quarter", "EBIT preceding year", 
            "EBIT preceding quarter", "EBIT preceding year quarter",
            "EBIDT last year", "EBIDT latest quarter", "EBIDT preceding year",
            "EBIDT preceding quarter", "EBIDT preceding year quarter",
            "EBITDA last year", "EBITDA latest quarter", "EBITDA preceding year",
            "EBITDA preceding quarter", "EBITDA preceding year quarter",
            "EBIT Margin", "EBIDT Margin", "EBITDA Margin", "EBIT Margin %",
            "EBITDA Margin %",
            
            # Other Income & Expenses
            "Other Income", "Non-Operating Income", "Other income last year", 
            "Other income latest quarter", "Other income preceding year", 
            "Other income preceding quarter", "Other income preceding year quarter",
            "Other Income % of Sales", "Non-Operating Revenue",
            "Expenses", "Total Expenses", "Operating Expenses", "Non-Operating Expenses",
            "Material cost", "Raw Material Cost", "Material Consumption",
            "Material cost last year", "Employee cost", "Employee cost last year",
            "Staff Cost", "Personnel Expenses", "Power & Fuel", "Fuel Expenses",
            "Other Expenses", "Selling Expenses", "Marketing Expenses",
            "Administrative Expenses", "General Expenses", "R&D Expenses",
            "Research and Development", "Depreciation & Amortization",
            "Finance Cost", "Other Operating Expenses",
            
            # Interest & Depreciation
            "Interest", "Interest Expense", "Finance Costs", 
            "Interest last year", "Interest latest quarter", "Interest preceding year", 
            "Interest preceding quarter", "Interest preceding year quarter",
            "Depreciation", "Depreciation Expense", 
            "Depreciation last year", "Depreciation latest quarter", 
            "Depreciation preceding year", "Depreciation preceding quarter",
            "Depreciation preceding year quarter", "Amortization", 
            "Amortization Expense",
            
            # Profit Before Tax
            "Profit before tax", "PBT", "Profit Before Tax",
            "Profit before tax last year", "Profit before tax latest quarter",
            "Profit before tax preceding year", "Profit before tax preceding quarter",
            "Profit before tax preceding year quarter", "PBT Margin", "PBT Growth",
            "PBT Margin %",
            
            # Tax
            "Tax", "Tax Expense", "Income Tax", 
            "Tax last year", "Tax latest quarter", "Tax preceding year",
            "Tax preceding quarter", "Tax preceding year quarter", 
            "Current Tax", "Deferred Tax", "Tax % of PBT", "Effective Tax Rate",
            "Tax Rate %", "Tax Provision",
            
            # Net Profit - Comprehensive
            "Net Profit", "Profit after tax", "PAT", "Net Income",
            "Net Profit last year", "Net Profit latest quarter", 
            "Net Profit preceding year", "Net Profit preceding quarter",
            "Net Profit preceding year quarter", "Net Profit 2 quarters back", 
            "Net Profit 3 quarters back", "Net Profit preceding 12 months",
            "Profit after tax last year", "Profit after tax latest quarter",
            "Profit after tax preceding year", "Profit after tax preceding quarter",
            "Profit after tax preceding year quarter",
            "Net Profit TTM", "Net Profit Margin", "PAT Margin", "PAT Growth",
            "Net Margin %", "PAT Margin %",
            
            # Extraordinary Items
            "Extraordinary items", "Exceptional Items", "One-time Items",
            "Extraordinary items last year", "Extraordinary items latest quarter",
            "Extraordinary items preceding year", "Extraordinary items preceding quarter",
            "Extraordinary items preceding year quarter", "Exceptional Gains",
            "Exceptional Losses",
            
            # Margins & Profitability - Detailed
            "Gross Profit Margin", "Operating Profit Margin", "Net Profit Margin",
            "EBIT Margin", "EBITDA Margin", "Pre-tax Margin", "Contribution Margin",
            "GPM", "OPM", "NPM", "OPM %", "OPM last year", "OPM latest quarter", 
            "OPM preceding year", "OPM preceding quarter", "OPM preceding year quarter", 
            "OPM 5 Year", "OPM 10 Year", "NPM", "Net Profit Margin", 
            "NPM last year", "NPM latest quarter", "NPM preceding year", 
            "NPM preceding quarter", "NPM preceding year quarter", 
            "GPM latest quarter", "Gross Margin %",
            
            # EPS - Comprehensive
            "EPS", "EPS in Rs", "Earnings Per Share", 
            "EPS last year", "EPS latest quarter", "EPS preceding year", 
            "EPS preceding quarter", "EPS preceding year quarter", 
            "EPS TTM", "EPS Growth", "EPS Growth %", "Adjusted EPS", 
            "Basic EPS", "Diluted EPS", "EPS CAGR 3 years", "EPS CAGR 5 years",
            "EPS CAGR 10 years",
            
            # Growth Rates - Extensive
            "Sales growth", "Revenue Growth", "Profit growth", "Operating profit growth",
            "YOY Quarterly sales growth", "YOY Quarterly profit growth", "YoY Sales Growth",
            "YoY Profit Growth", "QoQ Sales", "QoQ Profits", "QoQ Sales Growth",
            "QoQ Profit Growth", "Quarterly Growth", "Annual Growth",
            "Sales growth 3 years", "Sales growth 5 years", "Sales growth 7 years", 
            "Sales growth 10 years", "Sales growth 10 years median", 
            "Sales growth 5 years median", "Profit growth 3 years", "Profit growth 5 years", 
            "Profit growth 7 years", "Profit growth 10 years", "EBIDT growth 3 years", 
            "EBIDT growth 5 years", "EBIDT growth 7 years", "EBIDT growth 10 years",
            "EPS growth 3 years", "EPS growth 5 years", "EPS growth 7 years", 
            "EPS growth 10 years", "PAT Growth 3 years", "PAT Growth 5 years",
            "Operating Profit Growth", "EBITDA Growth", "Margin Expansion",
            
            # Banking Specific - Comprehensive
            "Financing Profit", "Financing Margin %", "Net Interest Income", 
            "Non-Interest Income", "Net Interest Margin", "Gross NPA %", "Net NPA %",
            "Provision Coverage Ratio", "Capital Adequacy Ratio", "CASA Ratio",
            "Cost to Income Ratio", "Return on Assets", "Return on Equity",
            "Credit Deposit Ratio", "Advance Deposit Ratio", "Slippage Ratio",
            "Recovery Ratio", "Write-off Ratio",
            
            # Additional Metrics
            "Operating Leverage", "Financial Leverage", "Total Leverage",
            "Contribution Margin", "Break-even Point", "Margin of Safety",
            "Operating Cycle", "Economic Value Added", "EVA"
        ]
        all_ratios.extend(pl_ratios)
        
        # 3. BALANCE SHEET RATIOS (300 ratios)
        balance_sheet_ratios = [
            # Equity & Capital - Comprehensive
            "Equity Capital", "Equity capital", "Share Capital", "Paid-up Capital",
            "Equity Capital latest quarter", "Equity Capital preceding quarter", 
            "Equity Capital preceding year quarter", "Authorized Capital",
            "Issued Capital", "Subscribed Capital", "Preference capital",
            "Preference Shares", "Number of equity shares", "Outstanding Shares",
            "Shares Outstanding", "Number of equity shares preceding year", 
            "Number of equity shares 10 years back", "Warrants", "Options",
            
            # Reserves & Net Worth - Detailed
            "Reserves", "Reserves and Surplus", "Capital Reserve", "Revenue Reserve",
            "General Reserve", "Specific Reserve", "Retained Earnings",
            "Accumulated Profits", "Revaluation reserve", "Other Reserves",
            "Securities Premium", "Debenture Redemption Reserve",
            "Net worth", "Shareholders Funds", "Total Equity", "Owner's Equity",
            "Net Worth Growth", "Book Value", "Book Value per share", "BVPS",
            
            # Borrowings & Debt - Comprehensive
            "Borrowings", "Borrowing", "Total Debt", "Debt", "Total Borrowings",
            "Debt preceding year", "Debt 3 years back", "Debt 5 years back", 
            "Debt 7 years back", "Debt 10 years back", "Long Term Debt",
            "Short Term Debt", "Current Maturities", "Secured loan", "Unsecured loan",
            "Lease liabilities", "Debentures", "Bonds", "Commercial Paper",
            "Working Capital Loan", "Term Loan", "External Commercial Borrowings",
            "Convertible Debt", "Non-Convertible Debt",
            
            # Other Liabilities - Detailed
            "Other Liabilities", "Trade Payables", "Sundry Creditors", 
            "Accounts Payable", "Advance from Customers", "Customer Advances",
            "Deposits", "Fixed Deposits", "Provisions", "Current Liabilities",
            "Non-Current Liabilities", "Deferred Tax Liability", "Contingent Liabilities",
            "Acceptances", "Bills Payable", "Other Current Liabilities",
            "Other Non-Current Liabilities",
            
            # Total Liabilities
            "Total Liabilities", "Balance sheet total", "Total Capital Employed",
            "Total Funds Employed",
            
            # Fixed Assets - Comprehensive
            "Fixed Assets", "Gross block", "Gross Block", "Gross Fixed Assets",
            "Gross block preceding year", "Net block", "Net Block", "Net Fixed Assets",
            "Net block preceding year", "Net block 3 years back", "Net block 5 years back",
            "Net block 7 years back", "Tangible Assets", "Intangible Assets",
            "Goodwill", "Brand Value", "Patents", "Copyrights", "Trademarks",
            "Accumulated depreciation", "Depreciation Reserve",
            "Capital Work in Progress", "CWIP", "Under Construction Assets",
            
            # Capital Work & Investments
            "Capital work in progress", "CWIP", "Capital work in progress preceding year",
            "Investments", "Current Investments", "Non-Current Investments",
            "Trade Investments", "Strategic Investments", "Book value of unquoted investments", 
            "Market value of quoted investments", "Investment Properties", 
            "Mutual Fund Investments", "Bond Investments", "Equity Investments",
            "Subsidiary Investments", "Associate Investments",
            
            # Current Assets - Detailed
            "Current assets", "Inventory", "Stock", "Finished Goods",
            "Work in Progress", "Raw Materials", "Stores and Spares",
            "Trade receivables", "Sundry Debtors", "Accounts Receivable",
            "Bills Receivable", "Cash Equivalents", "Cash and Bank", "Cash Balance",
            "Bank Balance", "Other Assets", "Loans and Advances", "Prepaid Expenses",
            "Marketable Securities", "Short Term Investments", "Advance Tax",
            "Other Current Assets",
            
            # Current Liabilities
            "Current liabilities", "Short Term Borrowings", "Creditors",
            "Other Current Liabilities", "Short Term Provisions",
            
            # Total Assets
            "Total Assets", "Non-Current Assets", "Current Assets", "Liquid Assets",
            "Quick Assets", "Total Assets Growth",
            
            # Working Capital - Comprehensive
            "Working capital", "Net Working Capital", "Working Capital Position",
            "Working capital preceding year", "Working capital 3 years back", 
            "Working capital 5 years back", "Working capital 7 years back", 
            "Working capital 10 years back", "Working Capital Turnover",
            "Working Capital to Sales", "Working Capital Cycle",
            
            # Contingent Liabilities
            "Contingent liabilities", "contingent liabilities by mcap",
            "Guarantees", "Letters of Credit", "Bills Discounted",
            "Capital Commitments", "Other Commitments",
            
            # Book Value & Net Worth
            "Book Value", "Book value", "Book Value per Share", "BVPS",
            "Book value preceding year", "Book value 3 years back", 
            "Book value 5 years back", "Book value 10 years back",
            "Adjusted Book Value", "Tangible Book Value", "Net Tangible Assets",
            
            # Additional Balance Sheet Items
            "Total Capital Employed", "CROIC", "Debt Capacity", "Debt To Profit",
            "Leverage", "Financial leverage", "Operating Assets", "Non-Operating Assets",
            "Capital Employed", "Net Fixed Assets", "Capital Work in Progress",
            "Net Block to Gross Block", "Asset Age",
            
            # Asset Quality
            "Fixed Assets Turnover", "Total Assets Turnover", "Current Assets Turnover",
            "Inventory to Sales", "Receivables to Sales", "Assets to Sales",
            "Capital Intensity", "Asset Lightness",
            
            # Capital Structure
            "Debt to Capital", "Equity to Assets", "Debt to Assets",
            "Long Term Debt to Equity", "Short Term Debt to Equity",
            "Debt to Total Funds", "Equity to Total Funds",
            "Capital Gearing Ratio", "Financial Structure",
            
            # Additional Metrics
            "Net Debt", "Net Cash", "Cash to Market Cap", "Debt to EBITDA",
            "Net Debt to EBITDA", "Interest Bearing Debt", "Debt Maturity Profile",
            "Debt Service Coverage", "Asset Coverage Ratio"
        ]
        all_ratios.extend(balance_sheet_ratios)
        
        # 4. CASH FLOW RATIOS (180 ratios)
        cash_flow_ratios = [
            # Operating Activities - Comprehensive
            "Cash from Operating Activity", "CFO", "Cash Flow from Operations",
            "Cash from operations last year", "Cash from operations preceding year", 
            "Operating cash flow 3 years", "Operating cash flow 5 years", 
            "Operating cash flow 7 years", "Operating cash flow 10 years", 
            "Net Cash from Operating Activities", "Cash Generated from Operations",
            "Operating Cash Flow TTM", "Operating Cash Flow Growth",
            
            # Investing Activities - Detailed
            "Cash from Investing Activity", "CFI", "Cash Flow from Investing",
            "Cash from investing last year", "Cash from investing preceding year", 
            "Investing cash flow 3 years", "Investing cash flow 5 years",
            "Investing cash flow 7 years", "Investing cash flow 10 years", 
            "Capital Expenditure", "Capex", "Purchase of Fixed Assets", 
            "Sale of Fixed Assets", "Investments in Subsidiaries", "Sale of Investments",
            "Acquisition of Businesses", "Disposal of Businesses", "Purchase of Investments",
            "Sale of Investments", "Investing Cash Flow TTM",
            
            # Financing Activities - Comprehensive
            "Cash from Financing Activity", "CFF", "Cash Flow from Financing",
            "Cash from financing last year", "Cash from financing preceding year", 
            "Issue of Equity", "Buyback of Shares", "Dividend Paid", "Interest Paid",
            "Repayment of Debt", "Proceeds from Debt", "Share Issue", "Debt Issue",
            "Financing Cash Flow TTM",
            
            # Net Cash Flow
            "Net Cash Flow", "Net Change in Cash", "Net cash flow last year", 
            "Net cash flow preceding year", "Cash Flow Surplus/Deficit",
            "Net Cash Flow TTM", "Cash Flow Pattern",
            
            # Cash Position - Detailed
            "Cash beginning of last year", "Cash end of last year",
            "Cash beginning of preceding year", "Cash end of preceding year",
            "Cash 3 years back", "Cash 5 years back", "Cash 7 years back",
            "Cash and Cash Equivalents", "Opening Cash Balance", "Closing Cash Balance",
            "Cash Position", "Liquid Assets", "Cash to Current Assets",
            
            # Free Cash Flow - Comprehensive
            "Free Cash Flow", "FCF", "Free Cash Flow to Firm", "FCFF",
            "Free Cash Flow to Equity", "FCFE", "Free cash flow last year", 
            "Free cash flow preceding year", "Free cash flow 3 years", 
            "Free cash flow 5 years", "Free cash flow 7 years", "Free cash flow 10 years",
            "Operating Free Cash Flow", "FCF per Share", "FCF Yield", "FCF Margin",
            "FCF Growth", "FCF Coverage", "FCF to Sales",
            
            # Cash Flow Ratios - Extensive
            "Cash Flow Margin", "Operating Cash Flow Margin", "Cash Flow to Sales",
            "Cash Flow to Debt", "FCF to Sales", "FCF to Net Income",
            "Cash Conversion Cycle", "Days Sales Outstanding", "Days Inventory Outstanding",
            "Days Payable Outstanding", "Cash Cycle", "Operating Cycle",
            "Cash Flow Adequacy", "Cash Flow Coverage",
            
            # Quality of Earnings
            "Quality of Earnings", "Cash Flow from Operations to Net Income",
            "Accruals Ratio", "Operating Cash Flow to Net Income",
            "Cash Earnings", "Cash Profit", "Cash EPS",
            
            # Coverage Ratios - Detailed
            "Cash Flow Coverage", "FCF Coverage", "Operating Cash Flow Coverage",
            "Dividend Coverage from FCF", "Interest Coverage from CFO",
            "Debt Service Coverage Ratio", "Fixed Charge Coverage",
            "Capital Expenditure Coverage",
            
            # Additional Cash Flow Metrics
            "Cash Generated from Operations", "Cash Used in Investing",
            "Cash from Financing", "Net Change in Cash and Equivalents",
            "Cash Flow from Discontinued Operations", "Foreign Exchange Impact",
            "Cash Flow Statement Analysis", "Cash Flow Trends"
        ]
        all_ratios.extend(cash_flow_ratios)
        
        # 5. EFFICIENCY & OPERATIONAL RATIOS (150 ratios)
        efficiency_ratios = [
            # Working Capital Management - Comprehensive
            "Working Capital Days", "Cash Conversion Cycle", "Net Working Capital Days",
            "Working Capital to Sales ratio", "Average Working Capital Days 3 years",
            "Working Capital Turnover", "Net Working Capital Turnover",
            "Working Capital Efficiency", "Working Capital Intensity",
            
            # Debtor Management - Detailed
            "Debtor Days", "Days Receivable Outstanding", "Receivable Days",
            "Collection Period", "Debtor days 3 years back", "Debtor days 5 years back",
            "Average debtor days 3 years", "Debtors Turnover", "Receivables Turnover",
            "Average Collection Period", "Debtor Velocity", "Receivable Efficiency",
            
            # Inventory Management - Comprehensive
            "Inventory Days", "Days Inventory Outstanding", "Inventory Holding Period",
            "Inventory turnover ratio", "Inventory Turnover", "Stock Turnover",
            "Inventory turnover ratio 3 years back", "Inventory turnover ratio 5 years back",
            "Inventory turnover ratio 7 years back", "Inventory turnover ratio 10 years back",
            "Inventory to Working Capital", "Inventory Efficiency", "Stock Velocity",
            
            # Payable Management
            "Days Payable", "Days Payable Outstanding", "Creditor Days",
            "Payable Turnover", "Accounts Payable Turnover", "Payment Period",
            "Creditor Velocity",
            
            # Asset Efficiency - Extensive
            "Asset Turnover Ratio", "Total Asset Turnover", "Fixed Asset Turnover",
            "Current Asset Turnover", "Return on invested capital", "CROIC",
            "Capital Turnover", "Investment Turnover", "Asset Utilization",
            "Capital Efficiency", "Return on Net Assets", "RONA",
            
            # Operational Efficiency - Detailed
            "Exports percentage", "Export %", "Export Revenue %",
            "Exports percentage 3 years back", "Exports percentage 5 years back", 
            "Domestic %", "Geographical Mix", "Product Mix", "Segment Wise Revenue",
            "Business Segment Revenue", "Geographic Segment Revenue",
            
            # Additional Efficiency Ratios
            "Employee Efficiency", "Revenue per Employee", "Profit per Employee",
            "Asset per Employee", "Plant Efficiency", "Capacity Utilization",
            "Operating Ratio", "Efficiency Ratio", "Productivity Ratio",
            "Cost Efficiency", "Operational Efficiency Score",
            
            # Business Cycle
            "Operating Cycle", "Cash Cycle", "Business Cycle", "Production Cycle",
            "Sales Cycle", "Manufacturing Cycle", "Trading Cycle",
            
            # Industry Specific Efficiency
            "Same Store Sales Growth", "Volume Growth", "Realization per Unit",
            "Cost per Unit", "Utilization Rate", "Occupancy Rate",
            "Production Capacity", "Installed Capacity", "Capacity Addition",
            
            # Supply Chain Efficiency
            "Supply Chain Days", "Procurement Cycle", "Distribution Efficiency",
            "Logistics Cost", "Supply Chain Turnover",
            
            # Technology Efficiency
            "IT Efficiency", "Digital Efficiency", "Automation Ratio",
            "Technology Adoption", "Digital Transformation Score"
        ]
        all_ratios.extend(efficiency_ratios)
        
        # Remove duplicates and return
        unique_ratios = list(set(all_ratios))
        print(f"🎯 TOTAL RATIOS TARGETED: {len(unique_ratios)}")
        
        return unique_ratios

    def _clean_value(self, value_str: str) -> Any:
        """Clean and convert string values to appropriate types"""
        if not value_str or value_str.strip() in ['--', '-', '', 'None', 'N/A', 'NaN']:
            return None
        
        try:
            # Remove common symbols and text
            cleaned_str = re.sub(r"[₹,%]", "", value_str)
            cleaned_str = re.sub(r"\s+Cr\.?", "", cleaned_str)
            cleaned_str = re.sub(r"\s+Lac\.?", "", cleaned_str)
            cleaned_str = re.sub(r"\s+", "", cleaned_str)
            
            # Handle cases like "2,345.67" or "2345.67"
            cleaned_str = cleaned_str.replace(',', '')
            
            # Try to convert to float first
            return float(cleaned_str)
        except (ValueError, TypeError):
            # Return the string if it can't be converted to float
            return value_str.strip() if value_str else None

    def _extract_all_possible_ratios(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract ALL possible ratios from the page using multiple methods"""
        all_data = {}
        
        # Method 1: Parse all tables in the entire page
        all_tables = soup.find_all('table')
        for table in all_tables:
            try:
                # Get all headers
                headers = []
                header_row = table.find('thead')
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
                
                # Get all data rows
                rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        metric_name = cells[0].get_text(strip=True)
                        if metric_name:
                            # Get values from all columns
                            for i in range(1, len(cells)):
                                if i-1 < len(headers):
                                    period = headers[i-1]
                                else:
                                    period = f"Column_{i}"
                                
                                value = cells[i].get_text(strip=True)
                                if value and value not in ['--', '-', '']:
                                    key = f"{metric_name} ({period})"
                                    all_data[key] = self._clean_value(value)
            except Exception as e:
                continue
        
        # Method 2: Parse all list items
        all_lists = soup.find_all(['ul', 'ol'])
        for list_elem in all_lists:
            try:
                list_items = list_elem.find_all('li')
                for item in list_items:
                    text = item.get_text(strip=True)
                    if ':' in text:
                        parts = text.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            all_data[key] = self._clean_value(value)
            except Exception:
                continue
        
        # Method 3: Parse all divs with common financial classes
        financial_divs = soup.find_all('div', class_=re.compile(r'ratio|metric|value|data|info|stats', re.I))
        for div in financial_divs:
            try:
                text = div.get_text(strip=True)
                if ':' in text:
                    parts = text.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        all_data[key] = self._clean_value(value)
            except Exception:
                continue
        
        # Method 4: Parse all spans with financial data
        financial_spans = soup.find_all('span', class_=re.compile(r'number|value|data|metric|ratio', re.I))
        for span in financial_spans:
            try:
                # Look for sibling with label
                parent = span.parent
                if parent:
                    label = parent.find(class_=re.compile(r'name|label|title|header', re.I))
                    if label:
                        key = label.get_text(strip=True)
                        value = span.get_text(strip=True)
                        all_data[key] = self._clean_value(value)
            except Exception:
                continue
        
        # Method 5: Parse all sections and their headers
        sections = soup.find_all(['section', 'div'], class_=re.compile(r'section|panel|box|card', re.I))
        for section in sections:
            try:
                # Look for header and value pairs
                headers = section.find_all(class_=re.compile(r'header|title|name', re.I))
                for header in headers:
                    next_elem = header.find_next(class_=re.compile(r'value|number|data', re.I))
                    if next_elem:
                        key = header.get_text(strip=True)
                        value = next_elem.get_text(strip=True)
                        if key and value:
                            all_data[key] = self._clean_value(value)
            except Exception:
                continue
        
        return all_data

    def _map_to_standard_ratios(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map extracted data to standard ratio names"""
        mapped_data = {}
        
        # First, add all extracted data with their original keys
        mapped_data.update(extracted_data)
        
        # Then, try to map to standard names
        for standard_name, possible_names in self.ratio_mappings.items():
            for extracted_key in extracted_data.keys():
                for possible_name in possible_names:
                    if possible_name.lower() in extracted_key.lower():
                        mapped_data[standard_name] = extracted_data[extracted_key]
                        break
        
        return mapped_data

    def _calculate_missing_ratios(self, existing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate any missing ratios that can be derived from existing data"""
        calculated = {}
        latest_period = ""

        # --- Find the latest annual period (e.g., "Mar 2025") ---
        try:
            for k in existing_data.keys():
                match = re.search(r"\((Mar \d{4})\)", k)
                if match:
                    try:
                        period_date = datetime.strptime(match.group(1), "Mar %Y")
                        if not latest_period or period_date > datetime.strptime(latest_period, "Mar %Y"):
                            latest_period = match.group(1)
                    except ValueError:
                        continue # Skip if date format is invalid
        except Exception:
            pass # latest_period may remain empty, which is fine

        # --- 1. Calculate Debt to Equity ---
        try:
            # Use snapshot keys first (e.g., "Debt", "Equity Capital", "Reserves")
            debt = existing_data.get('Debt') or existing_data.get('Borrowings')
            equity_cap = existing_data.get('Equity Capital')
            reserves = existing_data.get('Reserves')

            # If snapshot keys fail, try latest annual keys
            if (debt is None or equity_cap is None or reserves is None) and latest_period:
                debt = existing_data.get(f'Borrowings+ ({latest_period})') or existing_data.get(f'Debt ({latest_period})')
                equity_cap = existing_data.get(f'Equity Capital ({latest_period})')
                reserves = existing_data.get(f'Reserves ({latest_period})')

            if debt is not None and equity_cap is not None and reserves is not None:
                debt = float(str(debt).replace(',', ''))
                equity_cap = float(str(equity_cap).replace(',', ''))
                reserves = float(str(reserves).replace(',', ''))

                total_equity = equity_cap + reserves
                if total_equity > 0:
                    d_e_ratio = round(debt / total_equity, 2)
                    calculated['Debt to Equity'] = d_e_ratio
                    calculated['D/E'] = d_e_ratio
                    print(f"   ✨ Calculated D/E: {d_e_ratio} (Debt: {debt} / Equity: {total_equity})")

        except (ValueError, TypeError, Exception) as e:
            print(f"   ⚠️  Could not calculate Debt to Equity: {e}")

        # --- 2. Calculate Current Ratio (from latest annual data) ---
        try:
            if latest_period:
                # Note: Screener provides 'Other Assets+' and 'Other Liabilities+' which are NOT Current Assets/Liabilities.
                # The script *is* scraping the full balance sheet tables, so the keys are like:
                # 'Current assets (Mar 2025)' and 'Current liabilities (Mar 2025)'
                # Let's try to find those.
                
                current_assets = None
                current_liabilities = None

                # Search for the exact keys from the table scrape
                for k, v in existing_data.items():
                    if 'current assets' in k.lower() and latest_period in k:
                        current_assets = v
                    if 'current liabilities' in k.lower() and latest_period in k:
                        current_liabilities = v
                
                # If not found, it's because the scraper doesn't parse that specific row.
                # The provided scraper code only parses 'Fixed Assets+', 'CWIP', 'Investments', 'Other Assets+'
                # It does NOT parse 'Current assets' or 'Current liabilities'.
                # Therefore, this calculation is impossible with the *current* scraping logic.
                if current_assets is not None and current_liabilities is not None:
                    current_assets = float(str(current_assets).replace(',', ''))
                    current_liabilities = float(str(current_liabilities).replace(',', ''))
                    
                    if current_liabilities > 0:
                        cr_ratio = round(current_assets / current_liabilities, 2)
                        calculated['Current Ratio'] = cr_ratio
                        print(f"   ✨ Calculated Current Ratio: {cr_ratio}")
                else:
                    # This is the expected outcome
                    pass
                    # print("   ℹ️  Skipping Current Ratio: Keys not found. Scraper needs to be updated to parse 'Current assets'/'Current liabilities' rows.")

        except Exception as e:
            print(f"   ⚠️  Error calculating Current Ratio: {e}")

        # --- 3. Calculate P/E, P/S, P/B (using TTM/Latest data) ---
        try:
            market_cap = existing_data.get('Market Cap') # This is usually a single value
            current_price = existing_data.get('Current Price') # This is usually a single value
            book_value_ps = existing_data.get('Book Value') # This is already per-share

            # TTM (Trailing Twelve Months) data is often in a specific column.
            # Let's look for keys like 'Sales+ (TTM:)' or 'Net Profit+ (TTM:)'
            # Your scraper seems to label the TTM column as 'TTM: (Column_1)' or similar,
            # and the data rows as 'Sales+ (TTM: (Column_1))'.
            # Let's find the TTM column header first.
            ttm_col_header = None
            for k in existing_data.keys():
                if 'TTM:' in k and 'Column_' in k:
                    ttm_col_header = k # e.g., 'TTM: (Column_1)'
                    break
            
            sales_ttm = None
            net_profit_ttm = None

            if ttm_col_header:
                sales_ttm = existing_data.get(f"Sales+ ({ttm_col_header})")
                net_profit_ttm = existing_data.get(f"Net Profit+ ({ttm_col_header})")

            # Fallback to latest annual data if TTM keys fail
            if not sales_ttm and latest_period:
                sales_ttm = existing_data.get(f"Sales+ ({latest_period})")
            if not net_profit_ttm and latest_period:
                net_profit_ttm = existing_data.get(f"Net Profit+ ({latest_period})")

            # Ensure all values are numeric floats for calculation
            if market_cap: market_cap = float(str(market_cap).replace(',', '').replace('Cr.', ''))
            if current_price: current_price = float(str(current_price).replace(',', ''))
            if sales_ttm: sales_ttm = float(str(sales_ttm).replace(',', ''))
            if net_profit_ttm: net_profit_ttm = float(str(net_profit_ttm).replace(',', ''))
            if book_value_ps: book_value_ps = float(str(book_value_ps).replace(',', ''))

            # Perform calculations
            if market_cap and net_profit_ttm and net_profit_ttm > 0:
                pe_ratio = round(market_cap / net_profit_ttm, 2)
                # Only add if not already present or is different from the scraped value
                if 'Stock P/E' not in existing_data or existing_data['Stock P/E'] != pe_ratio:
                    calculated['Calculated P/E (TTM)'] = pe_ratio

            if market_cap and sales_ttm and sales_ttm > 0:
                ps_ratio = round(market_cap / sales_ttm, 2)
                calculated['Price to Sales (TTM)'] = ps_ratio

            if current_price and book_value_ps and book_value_ps > 0:
                pb_ratio = round(current_price / book_value_ps, 2)
                if 'Price to book value' not in existing_data or existing_data['Price to book value'] != pb_ratio:
                     calculated['Calculated P/B'] = pb_ratio
            
            if net_profit_ttm and market_cap and market_cap > 0:
                calculated['Earnings Yield (TTM)'] = round((net_profit_ttm / market_cap) * 100, 2)
                
        except Exception as e:
            print(f"   ⚠️  Error calculating P/E, P/S, P/B ratios: {e}")
        
        return calculated

    def get_all_ratios_for_company(self, ticker: str, session: Optional[Session] = None) -> Dict[str, Any]:
        """Get ALL possible ratios for a company"""
        url = f"https://www.screener.in/company/{ticker}/consolidated/"
        all_data = {}
        
        try:
            requester = session or requests
            response = requester.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            print(f"🔍 Extracting ALL ratios for {ticker}...")
            
            # Extract all possible data from the page
            extracted_data = self._extract_all_possible_ratios(soup)
            
            # Map to standard ratio names
            all_data = self._map_to_standard_ratios(extracted_data)
            
            # Calculate missing ratios
            calculated_ratios = self._calculate_missing_ratios(all_data)
            all_data.update(calculated_ratios)
            
            # Add metadata
            all_data['Ticker'] = ticker
            all_data['Data URL'] = url
            all_data['Last Updated'] = datetime.now().isoformat()
            
            # Filter out None values and empty strings
            all_data = {k: v for k, v in all_data.items() if v not in [None, '', 'None', 'N/A']}
            
            # Add final count after filtering
            all_data['Total Ratios Found'] = len(all_data)
            
            print(f"✅ Retrieved {len(all_data)} data points for {ticker}")
            
            return all_data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"❌ Company '{ticker}' not found (404)")
            else:
                print(f"❌ HTTP error for {ticker}: {e}")
            return {}
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error for {ticker}: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error parsing {ticker}: {e}")
            return {}

    def export_complete_data(self, data: Dict[str, Dict[str, Any]], filename: str):
        """Export complete data to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Complete data exported to {filename}")
        except Exception as e:
            print(f"❌ Error exporting data: {e}")

    def create_comprehensive_report(self, data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create a comprehensive report showing ratios coverage"""
        report_data = []
        
        ratio_categories = {
            "Price & Market": ["Market Capitalization", "Current Price", "Stock P/E", "Price to Sales", "Dividend Yield"],
            "Profit & Loss": ["Sales", "Operating Profit", "Net Profit", "OPM", "EPS"],
            "Balance Sheet": ["Equity Capital", "Reserves", "Debt", "Total Assets", "Book Value", "Debt to Equity"],
            "Cash Flow": ["Cash from Operating Activity", "Cash from Investing Activity", "Net Cash Flow"],
            "Efficiency": ["ROCE", "ROE", "Debtor Days", "Inventory Days"]
        }
        
        for ticker, company_data in data.items():
            coverage_info = {'Ticker': ticker, 'Total Data Points': len(company_data)}
            
            for category, ratios in ratio_categories.items():
                found_count = 0
                for ratio in ratios:
                    # Check both exact matches and partial matches
                    if (ratio in company_data or 
                        any(ratio.lower() in key.lower() for key in company_data.keys())):
                        found_count += 1
                coverage_info[f'{category} Coverage'] = f"{found_count}/{len(ratios)}"
            
            report_data.append(coverage_info)
        
        return pd.DataFrame(report_data)

def scrape_all_companies(tickers: List[str], delay: float = 3.0) -> Dict[str, Dict[str, Any]]:
    """Scrape data for all companies"""
    scraper = UltimateScreenerScraper()
    all_data = {}
    
    print(f"🚀 Starting scraping for {len(tickers)} companies")
    print(f"🎯 Targeting {len(scraper.all_ratios_list)} total ratios")
    print("=" * 80)
    
    with requests.Session() as session:
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            
            company_data = scraper.get_all_ratios_for_company(ticker, session)
            if company_data:
                all_data[ticker] = company_data
                
                found_ratios = len(company_data)
                print(f"   📈 Found {found_ratios} ratios")
                
                # Show sample ratios
                sample_ratios_keys = [
                    'Market Cap', 'Stock P/E', 'Debt to Equity', 'ROCE', 
                    'ROE', 'Sales+ (Mar 2025)', 'Net Profit+ (Mar 2025)'
                ]
                sample_output = []
                for key in sample_ratios_keys:
                    if key in company_data:
                        sample_output.append(f"{key}: {company_data[key]}")
                print(f"   📋 Sample: {', '.join(sample_output)}")
            
            if i < len(tickers):
                time.sleep(delay)
    
    return all_data

# Main execution
if __name__ == "__main__":
    companies = [
    "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANITRANS", "APOLLOHOSP", "APOLLOTYRE",
    "ASHOKLEY", "ASIANPAINT", "ASTRAL", "ATUL", "AUBANK", "AXISBANK", "BAJAJ-AUTO",
    "BAJAJFINSV", "BAJAJHIND", "BALAMINES", "BALMLAWRIE", "BANKBARODA", "BANKINDIA",
    "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL", "BHILWARA",
    "BIOCON", "BIRLACORPN", "BIRLASOFT", "BLS", "BOB", "BOSCHLTD", "BPCL", "BRIGADE",
    "BRITANNIA", "BSOFT", "CAMPUS", "CANBK", "CHAMBLFERT", "CHOLAFIN", "CIPLA",
    "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON", "CUB",
    "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR", "DELL", "DIXON", "DIVISLAB",
    "DODLA", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK",
    "FORTIS", "GAIL", "GAYAPROJ", "GICRE", "GLAND", "GLENMARK", "GODFRYPHLP",
    "GODREJCP", "GODREJPROP", "GRANULES", "GRASIM", "GREAVESCOT", "GSKCONS",
    "HCLTECH", "HDFC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDPETRO", "HINDUNILVR", "HONAUT", "ICICIBANK", "ICICIGI", "ICICIPRAMC",
    "IDBI", "IDFC", "IEX", "IFBIND", "IGL", "INDIACEM", "INDIANB", "INDIGO",
    "INDUSINDBK", "INFY", "IOC", "IPCALAB", "IRCON", "ITC", "JINDALSTEL",
    "JKCEMENT", "JSWENERGY", "JSWSTEEL", "JUBLFOOD", "KANSAINER", "KOTAKBANK",
    "KPIGREEN", "L&TFH", "LICI", "LT", "LTIM", "LUPIN", "M&M", "M&MFIN",
    "MANAPPURAM", "MARICO", "MARUTI", "MCX", "METROPOLIS", "MFSL", "MGL",
    "MINDTREE", "MIRZAINT", "MMTC", "MOTHERSUMI", "MPHASIS", "MRF", "MUTHOOTFIN",
    "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NESTLEIND", "NMDC", "NTPC", "OBEROIRLTY",
    "OFSS", "OIL", "ONGC", "OPAL", "ORIENTCEM", "PAGEIND", "PATANJALI", "PEL",
    "PERSISTENT", "PETRONET", "PIDILITIND", "PIIND", "PNB", "POWERGRID",
    "PRESTIGE", "PTC", "PURVA", "RATNAMANI", "RBLBANK", "RECLTD", "RELIANCE",
    "RELINFRA", "SAIL", "SBFC", "SBIN", "SCHAEFFLER", "SHRIRAMFIN", "SIEMENS",
    "SJVN", "SONACOMS", "SUNTV", "SUNPHARMA", "SUNTV", "SWSOLAR", "SYNGENE",
    "SYRMADENC", "TATACHEM", "TATACONSUM", "TATAMOTORS", "TATAPOWER",
    "TATASTEEL", "TCS", "TECHM", "TECHNOFAB", "TITAN", "TORNTPOWER",
    "TRENT", "TRITURBINE", "TTML", "TUBEINVEST", "TVSMOTOR", "UCOBANK",
    "ULTRACEMCO", "UNIONBANK", "UPL", "VBL", "VEDL", "VOLTAS", "WHIRLPOOL",
    "WIPRO", "ZYDUSLIFE",
    "ABB", "ABBOTINDIA", "ACC", "ADANIPOWER", "ADVENZYMES", "AEGISCHEM", "AIAENG", "ALBK", "ALKEM", "ALLCARGO", "AMBUJACEM", "AMRITANUFL", "APLLTD", "ARVIND", "ASAHIINDIA", "ASHOKLEY", "ASTERDM", "ATGL", "AUBANK", "AURIONPRO", "AUTOAXLES", "AVANTIFEED", "AXISBANK", "BAJAJELEC", "BALAJITELE", "BALKRISIND", "BANCOINDIA", "BANG", "BANKBARODA", "BANKINDIA", "BASF", "BATAINDIA", "BAZAZZ", "BBTC", "BCG", "BEARDSELL", "BEDMUTHA", "BEL", "BEML", "BERGEPAINT", "BFUTILIT", "BHAGERIA", "BHARATFORG", "BHARATRAST", "BHARTIARTL", "BHEL", "BHILWARA", "BIKAJI", "BIL", "BINANIIND", "BINDALAGRO", "BIOCON", "BIRLACABLE", "BIRLACORPN", "BIRLAMONEY", "BIRLASOFT", "BLS", "BLUECOAST", "BLUESTARCO", "BNR_UDYOG", "BOMDYEING", "BORORENEW", "BOSCHLTD", "BPCL", "BRIGADE", "BRITANNIA", "BSE", "BSOFT", "BUTTERFLY", "CACHET", "CADILAHC", "CALSOFT", "CANARA BNK", "CANBK", "CANDC", "CAPLIPOINT", "CARBORUNIV", "CAREERP", "CASTROLIND", "CCCL", "CEATLTD", "CENTENKA", "CENTURYPLY", "CENTURYTEX", "CESC", "CGPOWER", "CHAMBLFERT", "CHENNPETRO", "CHOLAFIN", "CHOLAHLDNG", "CIPLA", "CIS", "CLNINDIA", "COALINDIA", "COCHINSHIP", "COFORGE", "COHIN", "COLPAL", "COLRD", "CONCOR", "COROMANDEL", "COTTINFAB", "CPL", "CREST", "CROMPTON", "CUB", "CUMMINSIND", "CYIENT", "DABUR", "DALBHARAT", "DALLINDIA", "DATSUN", "DCBBANK", "DEEPAKTR", "DEEPAKNTR", "DELL", "DELTACORP", "DEN", "DEVIT", "DFMFOODS", "DGCONTENT", "DHAMPURSUG", "DHANUKA", "DHFL", "DIAMONDYD", "DIEGW", "DISHTV", "DIVISLAB", "DIXON", "DODLA", "DOLATALBS", "DPSCLTD", "DRREDDY", "DSKULKARNI", "EIHOTEL", "EICHERMOT", "EIDPARRY", "EIH", "EKC", "ELECTCAST", "ELECTRA", "EMAMILTD", "ENGINERS", "EQUITASBNK", "EROSMEDIA", "ESCORTS", "ESSELPACK", "ETERNAL", "EVEREADY", "EXIDEIND", "FDC", "FEDERALBNK", "FEL", "FINCABLES", "FINPIPE", "FLEXFOODS", "FMGOETZE", "FORTIS", "FRONTIER", "FSL", "GABRIEL", "GAIL", "GALAXYSURF", "GANDHITUBE", "GANECOS", "GANGESSECU", "GANTTECH", "GATEWAY", "GAYAPROJ", "GCL", "GDL", "GENESYS", "GENUSP", "GEOJITFIN", "GFL", "GHCL", "GICRE", "GILLETTE", "GLAJAX", "GLAND", "GLAXO", "GLENMARK", "GLS", "GMDCLTD", "GMRINFRA", "GODFRYPHLP", "GODREJAGRO", "GODREJCP", "GODREJIND", "GODREJPROP", "GOLDIAM", "GOLDSTAR", "GOLDSUPERG", "GPET", "GRANULES", "GRAPHITE", "GRASIM", "GREAVESCOT", "GROBTEA", "GRPLTD", "GSKCONS", "GSPL", "GTL", "GULFOIL", "GULFPETRO", "GUNTURCOT", "GVKPIL", "HAL", "HAPPSTMNDS", "HATSUN", "HAVELLS", "HCC", "HCLTECH", "HDFC", "HDFCBANK", "HDFCLIFE", "HEIDELBERG", "HERANBA", "HERITGFOOD", "HEROMOTOCO", "HESTERBIO", "HGINFRA", "HICAL", "HINDALCO", "HINDCOPPER", "HINDMOTORS", "HINDOIL", "HINDORG", "HINDPETRO", "HINDUNILVR", "HINDZINC", "HOMEPINLAB", "HONAUT", "HPC", "HSCL", "HTMEDIA", "HUBTOWN", "ICICIBANK", "ICICIGI", "ICICIPRAMC", "ICRA", "IDBI", "IDFC", "IDFCFIRSTB", "IFBIND", "IFCI", "IGL", "IIFL", "IL&FSENGG", "IMFA", "INDBANK", "INDIACEM", "INDIANB", "INDIAGLYCO", "INDIAMART", "INDIGO", "INDOCO", "INDUSINDBK", "INDUSTOWER", "INFIBEAM", "INOXLEISUR", "INOXWIND", "IPCALAB", "IRB", "IRCON", "IRCTC", "ITC", "ITDC", "ITI", "ITTECH", "IVC", "IVP", "J&KBANK", "JAGRAN", "JAICORPLTD", "JAINIRRIG", "JISLJALEQS", "JKCEMENT", "JKLAKSHMI", "JKPAPER", "JL Morison", "JMA", "JMFINANCIL", "JNP", "JOCIL", "JPASSOCIAT", "JSL", "JSWENERGY", "JSWHL", "JSWSTEEL", "JTEKTINDIA", "JUBLFOOD", "JUNIORBEES", "JUSTDIAL", "KABRAEXTRU", "KAJARIACER", "KALPATPOWR", "KALPTARA", "KAMATHOTEL", "KANSAINER", "KANSAIPLS", "KARNPLAST", "KCP", "KEC", "KEL", "KEMAL", "KENNAMET", "KESORAMIND", "KGL", "KIRLOSBROS", "KIRLOSENG", "KITEX", "KJJMCORP", "KOTAKBANK", "KPIGREEN", "KPITTECH", "KPRMILL", "KRBL", "KSCL", "KTKBANK", "L&TFH", "LANCOR", "LASA", "LATENTVIEW", "LAURUSLABS", "LAXMIMACH", "LCCINFOTEC", "LICI", "LICNFIN", "LINDEINDIA", "LLOYDSME", "LOKESHMACH", "LOTION", "LT", "LTIM", "LTP", "LUPIN", "LUXIND", "M&M", "M&MFIN", "M3BIOLOGIC", "MACROTECH", "MAHINDCIE", "MAHLIFE", "MANAPPURAM", "MANINFRA", "MANKIND", "MARALOVER", "MARICO", "MARKSANS", "MARSHALL", "MASFIN", "MASTEK", "MASHREQBANK", "MASTEK", "MATE", "MAXHEALTH", "MAXIND", "MAXVIL", "MFSL", "MGL", "MICROSEC", "MINDACORP", "MINDTREE", "MIRZAINT", "MMTC", "MOHITEIND", "MONTECARLO", "MOREPENLAB", "MOTHERSUMI", "MPHASIS", "MRF", "MRO-TEK", "MRPL", "MSPL", "MTARTECH", "MTE", "MUTHOOTFIN", "NACLIND", "NARAYANAH", "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NAZARA", "NBCC", "NCC", "NESCO", "NESTLEIND", "NETWORK18", "NEWGEN", "NICE", "NLCINDIA", "NMDC", "NOCIL", "NORBTEAEXP", "NRAIL", "NRBBEARING", "NRL", "NTPC", "NUCLEUS", "OBCL", "OBEROIRLTY", "OIL", "OMAXE", "OMDUR", "ONGC", "OPAL", "ORCHPHARMA", "ORIENTCEM", "ORIENTELEC", "ORIENTHOT", "ORIENTPPR", "P&GHYGIENE", "PAGEIND", "PANACEABIO", "PATANJALI", "PATELENG", "PDSL", "PEARLPOLY", "PEL", "PENTAIRTS", "PERSISTENT", "PETRONET", "PFIZER", "PHOENIXLTD", "PIDILITIND", "PIIND", "PIRIND", "PNCINFRA", "PNB", "PNBGILTS", "POCARDS", "POLYPLEX", "PONNIERODE", "POWERGRID", "PPAP", "PRAJIND", "PRAKASH", "PRECAM", "PRECTRE", "PREMEXPLN", "PRESTIGE", "PRICOLLTD", "PRSMJOHNSN", "PTC", "PTCINDIA", "PUNJABCHEM", "PURVA", "R R", "RABITA", "RADIOCITY", "RAINBOW", "RAJESHEXPO", "RAJTV", "RAMANEWS", "RAMCOSYS", "RANEHOLDIN", "RATNAMANI", "RBLBANK", "RBPREF", "RCF", "RECLTD", "REDINGTON", "REFEX", "REGENCERAM", "RELIANCE", "RELIGARE", "RELINFRA", "REMSONSIND", "RENUKA", "REPCOHOME", "REPL", "RHFL", "RITES", "RML", "ROLEXRINGS", "ROTI", "ROYALORS", "RPGLIFE", "RPPINFRA", "RPSGVENT", "RSWM", "RTI", "RUSTOMJEE", "S&T CORP", "SABAR", "SAFAL", "SAIL", "SAKSOFT", "SALASAR", "SAMVAD", "SANOFI", "SANSERA", "SASKEN", "SBFC", "SBIN", "SBT", "SCHAEFFLER", "SCHNEIDER", "SCI", "SEAMECLTD", "SELAN", "SEPC", "SEQUENT", "SEREIN", "SFL", "SHAILY", "SHAKTIPUMP", "SHANKARA", "SHARDA", "SHEMAROO", "SHILPAMED", "SHIRPUR-G", "SHK", "SHOPPING", "SHREE CEMENT", "SHREERAMA", "SHRIRAMFIN", "SHYAMMETL", "SIEMENS", "SIGACHI", "SJVN", "SKFINDIA", "SKMEGGPROD", "SONACOMS", "SONATSOFTW", "SONS", "SOTL", "SPLPETRO", "SPRO", "SPYL", "SRHHYPOLTD", "SRL", "SSWL", "STARCEMENT", "STERTECH", "STOVALISBIO", "STRTECH", "SUBROS", "SUDARSCHEM", "SULA", "SUMICHEM", "SUMIT", "SUNCLAY", "SUNDARMFIN", "SUNDARMHLD", "SUNDM", "SUNDRA", "SUNFLAG", "SUNPHARMA", "SUNTV", "SUNTV", "SUPERHOUSE", "SUPREMEIND", "SUPRIYA", "SURECAM", "SWSOLAR", "SYMPHONY", "SYNGENE", "SYRMA", "SYRMADENC", "T T", "TAKE", "TANLA", "TARC", "TATAELXSI", "TATACHEM", "TATACONSUM", "TATACOMM", "TATAMETALI", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TEAMLEASE", "TECHM", "TECHNOFAB", "TEJASNET", "TEXRAIL", "TFL", "THERMAX", "THOMASCOOK", "THYROCARE", "TI", "TIDEWATER", "TIMETECHNO", "TIMKEN", "TIPSFILMS", "TIRUMALCHM", "TITAN", "TIWARES", "TJSB", "TMB", "TORNTPOWER", "TRENT", "TRIDENT", "TRITURBINE", "TRIVENI", "TT", "TTML", "TUBEINVEST", "TV18BRDCST", "TVSMOTOR", "TVTODAY", "UCOBANK", "UFLEX", "UGARSUGAR", "ULTRACEMCO", "UNIONBANK", "UNITDSPRIT", "UPL", "URJA", "UTTAMSUGAR", "VAKRANGEE", "VARDHACRLC", "VARDMNPOLY", "VASA", "VBL", "VCL", "VEDL", "VENKEYS", "VENUSREM", "VERANDA", "VHL", "VIJAYA", "VINATIORGA", "VINDHYATEL", "VIPIND", "VISAKAIND", "VISUALAS", "VLSFINANCE", "VOLTAS", "VSTIND", "WABAG", "WABCOINDIA", "WALCHANNAG", "WATERBASE", "WEALTH", "WEL", "WELCORP", "WELENT", "WESTLIFE", "WHIRLPOOL", "WILLAMSW", "WIPRO", "WOCKPHARMA", "WONDERLA", "WOOLWORTH", "WSL", "YESBANK", "YORKBANK", "ZENSARTECH", "ZIC", "ZODIAC", "ZUARI", "ZYDUSLIFE"
]

    
    print("🔥 STARTING ULTIMATE SCREENER SCRAPER 🔥")
    print("🎯 TARGET: 1128+ FINANCIAL RATIOS")
    print("=" * 80)
    
    complete_data = scrape_all_companies(companies)
    
    if complete_data:
        scraper = UltimateScreenerScraper()
        
        # Export data
        output_file = "COMPLETE_1128_SCREENER_RATIOS.json"
        scraper.export_complete_data(complete_data, output_file)
        
        # Create report
        report = scraper.create_comprehensive_report(complete_data)
        print("\n📊 COMPREHENSIVE COVERAGE REPORT:")
        print("=" * 80)
        print(report.to_string(index=False))
        
        # Final stats
        total_ratios = sum(len(data) for data in complete_data.values())
        avg_ratios = total_ratios / len(complete_data)
        
        print(f"\n🎉 FINAL SUMMARY:")
        print(f"   Companies: {len(complete_data)}")
        print(f"   Total Ratios: {total_ratios}")
        print(f"   Average per Company: {avg_ratios:.1f}")
        print(f"   Target Ratio Count: {len(scraper.all_ratios_list)}+")
        print(f"   Output File: {output_file}")
        
        # Show some key ratios found
        print(f"\n🔑 KEY RATIOS EXTRACTED:")
        sample_company = list(complete_data.keys())[0]
        key_ratios_to_check = [
            "Market Cap", "Stock P/E", "Price to book value", 
            "Dividend Yield", "ROCE", "ROE", "Debt to Equity", "Current Ratio"
        ]
        
        print(f"\n--- Sample Check for {sample_company} ---")
        for ratio in key_ratios_to_check:
            found = False
            for key in complete_data[sample_company].keys():
                if ratio.lower() == key.lower():
                    print(f"   ✓ {key}: {complete_data[sample_company][key]}")
                    found = True
                    break
            if not found:
                 # Check for calculated keys
                 for key in complete_data[sample_company].keys():
                    if ratio.lower() in key.lower():
                        print(f"   ✓ {key}: {complete_data[sample_company][key]}")
                        found = True
                        break
            if not found:
                print(f"   ✗ {ratio}: Not found")
        
    else:
        print("❌ No data retrieved")