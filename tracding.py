import requests
import json
import base64
import urllib.parse
import time
from datetime import datetime, time as dt_time
import os
import jwt
import yfinance as yf
import re
import sys
import logging
import threading
from flask import Flask, request, abort

# --- Set up global logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()])

class KotakNeoAPI:
    def __init__(self, config_path='config.json'):
        self.config = self._load_config(config_path)
        if not self.config:
            logging.error(f"Error: Configuration file not found at '{config_path}'")
            self.access_token = None
            return

        self.session = requests.Session()
        self.access_token = None
        self.sid = None
        self.authtoken = None
        self.hs_server_id = None
        self.user_id = None
        self.headers = {}
        self.params = {}
        self.is_logged_in = False
        self.login_lock = threading.Lock()

    def _load_config(self, path):
        if not os.path.exists(path): return None
        with open(path, 'r') as f: return json.load(f)

    def login(self):
        with self.login_lock:
            if self.is_logged_in:
                logging.info("Login check: Already logged in.")
                return True

            logging.info("--- Starting Login Process ---")
            try:
                token_url = "https://napi.kotaksecurities.com/oauth2/token"
                code = f'{self.config["consumer_key"]}:{self.config["secret_key"]}'
                payload = {'grant_type': 'password', 'username': self.config["username"], 'password': self.config["password"]}
                headers = {"Authorization": f"Basic {base64.b64encode(bytes(code, 'utf-8')).decode('utf-8')}"}
                res = self.session.post(token_url, headers=headers, data=payload, timeout=10)
                res.raise_for_status()
                self.access_token = res.json()['access_token']
                
                validate_url = "https://gw-napi.kotaksecurities.com/login/1.0/login/v2/validate"
                payload = json.dumps({"mobileNumber": self.config["mobile_number"], "password": self.config["login_password"]})
                headers = {'Authorization': f'Bearer {self.access_token}', 'Content-Type': 'application/json'}
                res = self.session.post(validate_url, headers=headers, data=payload, timeout=10)
                res.raise_for_status()
                data = res.json()['data']
                self.authtoken = data['token']
                self.sid = data['sid']
                self.hs_server_id = data.get('hsServerId', '')
                
                totp_url = "https://gw-napi.kotaksecurities.com/login/1.0/login/v6/totp/validate"
                headers = {'sid': self.sid, 'Auth': self.authtoken, 'neo-fin-key': 'neotradeapi', 'Authorization': f'Bearer {self.access_token}', 'Content-Type': 'application/json'}
                payload = json.dumps({"mpin": "120620"})
                res = self.session.post(totp_url, headers=headers, data=payload, timeout=10)
                res.raise_for_status()
                data = res.json()['data']
                self.authtoken = data['token']
                self.sid = data['sid']
                self.hs_server_id = data.get('hsServerId', '')
                
                self.headers = {'accept': '*/*', 'sid': self.sid, 'Auth': self.authtoken, 'neo-fin-key': 'neotradeapi', 'Authorization': f'Bearer {self.access_token}'}
                self.params = {'sId': self.hs_server_id} if self.hs_server_id else {}
                
                logging.info("--- Login Successful. API is ready. ---")
                self.is_logged_in = True
                return True
            except Exception as e:
                logging.error(f"Login Error: {e}")
                return False

    def _make_request(self, method, url, headers, data=None, params=None):
        if not self.is_logged_in:
            logging.warning("Session appears closed. Attempting login...")
            if not self.login(): return None
            if isinstance(headers, dict): headers.update(self.headers)

        try:
            response = self.session.request(method, url, headers=headers, data=data, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logging.warning("Session expired (401). Resetting login state.")
                self.is_logged_in = False
            else:
                logging.error(f"HTTP Error {e.response.status_code}: {e}")
            return None
        except Exception as e:
            logging.error(f"Request Error: {e}")
            return None

    def get_quote(self, symbol):
        url = f"https://gw-napi.kotaksecurities.com/apim/quotes/1.0/quotes/neosymbol/{symbol}"
        headers = {**self.headers, 'Content-Type': 'application/json'}
        res = self._make_request('GET', url, headers=headers)
        if res and res.get('stat') == 'Ok' and 'data' in res:
            q = res['data']
            return {
                'ltp': float(q.get('ltp', 0)),
                'high_circuit': float(q.get('highCircuitLimit', 0)),
                'low_circuit': float(q.get('lowCircuitLimit', 0)),
                'tick_size': float(q.get('tckSz', 0.05))
            }
        return None

    @staticmethod
    def fetch_yfinance_price(symbol):
        try:
            yahoo_symbol = symbol.replace('-EQ', '.NS')
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty: return 0.0
            return float(hist['Close'].iloc[-1])
        except Exception:
            return 0.0

    def place_bracket_order(self, transaction_type, symbol, quantity, price, stop_loss_value, square_off_value, trailing_stop_loss=None, entry_type='limit', yfinance_price=0.0):
        logging.info(f"Placing BRACKET order: {transaction_type} {quantity} of {symbol}")
        
        quote = self.get_quote(symbol)
        base_price = float(price)
        
        if quote:
            tick = quote['tick_size']
            base_price = round(base_price / tick) * tick
            base_price = max(min(base_price, quote['high_circuit']), quote['low_circuit'])
        else:
            logging.warning("Quote failed. Using provided price without validation.")

        # STRICT ROUNDING FIX for "Tick Size" Error
        pr_str = f"{base_price:.2f}"
        
        pt = "L" 
        url = 'https://gw-napi.kotaksecurities.com/Orders/2.0/quick/order/rule/ms/place'
        jData = {
            "am": "NO", "dq": "0", "es": "nse_cm", "mp": "0", "pc": "BO", "pf": "N",
            "pr": pr_str, "pt": pt, "qt": str(quantity), "rt": "DAY", "tp": "0", 
            "ts": symbol, "tt": transaction_type.upper(), 
            "sot": "Absolute", "slt": "Absolute",
            "slv": str(stop_loss_value), "sov": str(square_off_value), "lat": "LTP",
            "tlt": "N", "tsv": "0", "ig": "", "ucc": self.config['ucc']
        }

        if trailing_stop_loss:
            jData['tlt'] = 'Y'; jData['tsv'] = str(trailing_stop_loss)

        payload = urllib.parse.urlencode({'jData': json.dumps(jData)}, quote_via=urllib.parse.quote)
        headers = {**self.headers, 'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8'}

        res = self._make_request('POST', url, headers=headers, data=payload, params=self.params)
        
        if res and res.get('stat') == 'Ok':
            logging.info(f"--> SUCCESS: Order Placed. No: {res.get('nOrdNo')}")
        else:
            logging.error(f"--> FAILED: {res.get('error', res.get('rejRsn', 'Unknown'))}")
        return res

# ==============================================================================
# | WEBHOOK MODE ONLY
# ==============================================================================
api_client = KotakNeoAPI(config_path='config.json')
app = Flask(__name__)

def handle_trade_signal(data):
    try:
        logging.info("--- SIGNAL RECEIVED ---")
        raw_action = data.get('action')
        raw_ticker = data.get('ticker', '').upper()
        webhook_price = float(data.get('price', 0))
        
        if not raw_action or not raw_ticker: return

        clean_ticker = raw_ticker.replace("NSE:", "").replace("BSE:", "")
        kotak_ticker = f"{clean_ticker}-EQ"
        logging.info(f"Sanitized Ticker: {kotak_ticker}")

        act = raw_action.lower()
        if act == 'buy': transaction_type = 'B'
        elif act == 'sell': transaction_type = 'S'
        else:
            logging.info(f" -> Ignoring Exit Signal ('{act}'). Bracket Order handles exits.")
            return

        if webhook_price <= 0:
            webhook_price = api_client.fetch_yfinance_price(kotak_ticker)
            if webhook_price <= 0:
                logging.error("No price available.")
                return

        if not api_client.login(): return
        config = api_client.config
        
        # 1. Quantity Logic
        quantity_map = config.get('quantity_map', {}) 
        if kotak_ticker in quantity_map:
            quantity = int(quantity_map[kotak_ticker])
            logging.info(f" -> Found in Config Map. Using Fixed Quantity: {quantity}")
        elif 'quantity' in data and float(data['quantity']) > 0:
            quantity = int(float(data['quantity']))
            logging.info(f" -> Not in Config. Using Strategy Quantity: {quantity}")
        else:
            quantity = config.get('webhook_default_qty', 1)
            logging.info(f" -> Using Default Quantity: {quantity}")
        
        # 2. Price Calculation with ROUNDING FIX
        TICK = 0.05
        buf = 0.002 
        raw_price = webhook_price * (1 + buf) if transaction_type == 'B' else webhook_price * (1 - buf)
        # Force rounding to 2 decimal places to satisfy Exchange
        limit_price = round(round(raw_price / TICK) * TICK, 2)
        
        # 3. SL/Target Logic
        sl_pts = config.get('webhook_stop_loss_points', 0)
        tgt_pts = config.get('webhook_target_points', 0)
        
        if sl_pts == 0:
            sl_pts = round(((webhook_price * config.get('webhook_stop_loss_percent', 0.5)) / 100) / TICK) * TICK
        if tgt_pts == 0:
            tgt_pts = round(((webhook_price * config.get('webhook_target_percent', 1.0)) / 100) / TICK) * TICK
            
        sl_pts = max(sl_pts, TICK)
        tgt_pts = max(tgt_pts, TICK)

        # 4. Place Order
        api_client.place_bracket_order(
            transaction_type=transaction_type,
            symbol=kotak_ticker,
            quantity=quantity,
            price=limit_price,
            stop_loss_value=f"{sl_pts:.2f}",
            square_off_value=f"{tgt_pts:.2f}",
            trailing_stop_loss=config.get('webhook_trailing_stop_loss'),
            yfinance_price=webhook_price
        )

    except Exception as e:
        logging.error(f"Signal Handler Error: {e}")

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.data.decode('utf-8'))
        passphrase = api_client.config.get('webhook_passphrase')
        if passphrase and data.get('passphrase') != passphrase: return "Unauthorized", 401
        threading.Thread(target=handle_trade_signal, args=(data,)).start()
        return "OK", 200
    except Exception: return "Error", 500

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].lower() != 'webhook':
        print("Usage: python trading_bot.py webhook")
    else:
        print("--- Starting in WEBHOOK Mode ---")
        if api_client.login(): app.run(host='0.0.0.0', port=5000)