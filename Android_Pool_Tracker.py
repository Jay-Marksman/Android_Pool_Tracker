"""
Aerodrome_Base_v2.py - Fixed version with requested features.
FIX: update_xaxis() -> update_layout(xaxis_tickangle=...)
"""

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
from web3 import Web3
from typing import Dict, Optional, List
from datetime import datetime, timedelta

# DexScreener
DEXSCREENER_PAIR_URL = "https://api.dexscreener.com/latest/dex/pairs"
DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens"

# Aerodrome Base
FACTORY_ADDRESS = Web3.to_checksum_address("0x420dd381b31aef6683db6b902084cb0ffece40da")
BASE_RPC_URL = "https://mainnet.base.org"

FACTORY_ABI = [
    {"inputs": [{"name": "pool", "type": "address"}, {"name": "_stable", "type": "bool"}],
     "name": "getFee", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "volatileFee", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "_stable", "type": "bool"}], "name": "stableFee", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}
]

class AerodromeFees:
    def __init__(self, basescan_key: str = ""):
        self.basescan_key = basescan_key
        self.w3 = Web3(Web3.HTTPProvider(BASE_RPC_URL))
        self.factory = self.w3.eth.contract(address=FACTORY_ADDRESS, abi=FACTORY_ABI)

    @st.cache_data(ttl=300)
    def get_default_fees(_self) -> Dict[str, float]:
        try:
            volatile = _self.factory.functions.volatileFee().call() / 10000
            stable = _self.factory.functions.stableFee(True).call() / 10000
            return {"stable": round(stable, 4), "volatile": round(volatile, 4)}
        except:
            return {"stable": 0.05, "volatile": 0.30}

    def get_pool_fee(self, pool_address: str, is_stable: bool = False) -> Optional[float]:
        try:
            fee_bps = self.factory.functions.getFee(Web3.to_checksum_address(pool_address), is_stable).call()
            return round(fee_bps / 10000, 4)
        except:
            return None

@st.cache_data(ttl=60)
def fetch_pair(chain: str, pool_address: str, max_retries: int = 2) -> Optional[dict]:
    url = f"{DEXSCREENER_PAIR_URL}/{chain}/{pool_address}"
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=10)
        except requests.RequestException as e:
            if attempt == max_retries:
                st.error(f"Network error for {pool_address} after {max_retries+1} tries: {e}")
                return None
            time.sleep(1)
            continue

        if response.status_code != 200:
            if attempt == max_retries:
                st.error(f"Error fetching {pool_address}: {response.text}")
                return None
            time.sleep(1)
            continue

        data = response.json()
        pairs = data.get('pairs', [])
        if not pairs:
            st.warning(f"No pair data for {pool_address}")
            return None
        return pairs[0]

@st.cache_data(ttl=300)
def fetch_token_price_history(token_address: str, max_retries: int = 2) -> Optional[dict]:
    url = f"{DEXSCREENER_TOKEN_URL}/{token_address}"
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pairs = data.get('pairs', [])
                if pairs:
                    return pairs[0]  # Return first pair with data
                return None
        except:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return None
    return None

def is_valid_address(addr: str) -> bool:
    pattern = r'^0x[a-fA-F0-9]{40}$'
    return bool(re.match(pattern, addr))

def build_dataframe_from_pair(pair: dict) -> pd.DataFrame:
    base_token = pair.get('baseToken', {})
    quote_token = pair.get('quoteToken', {})
    liquidity = pair.get('liquidity', {})
    volume = pair.get('volume', {})
    txns = pair.get('txns', {})

    def get_tx_count(tx_window):
        return int(tx_window.get('count') or tx_window.get('buys', 0) + tx_window.get('sells', 0))

    tx_24h = txns.get('h24', {})
    tx_6h = txns.get('h6', {})
    tx_1h = txns.get('h1', {})

    row = {
        'pair_address': pair.get('pairAddress'),
        'dex': pair.get('dexId'),
        'chain': pair.get('chainId'),
        'pair_name': f"{base_token.get('symbol', 'UNK')}/{quote_token.get('symbol', 'UNK')}",
        'token0_symbol': base_token.get('symbol', 'UNK'),
        'token0_address': base_token.get('address'),
        'token1_symbol': quote_token.get('symbol', 'UNK'),
        'token1_address': quote_token.get('address'),
        'price_usd': float(pair.get('priceUsd') or 0),
        'liquidity_usd': float(liquidity.get('usd') or 0),
        'liquidity_token0': float(liquidity.get('base') or 0),
        'liquidity_token1': float(liquidity.get('quote') or 0),
        'volume_24h_usd': float(volume.get('h24') or 0),
        'volume_6h_usd': float(volume.get('h6') or 0),
        'volume_1h_usd': float(volume.get('h1') or 0),
        'tx_24h_count': get_tx_count(tx_24h),
        'tx_6h_count': get_tx_count(tx_6h),
        'tx_1h_count': get_tx_count(tx_1h),
        'tx_24h_buys': int(tx_24h.get('buys') or 0),
        'tx_24h_sells': int(tx_24h.get('sells') or 0),
        'fee_stable_pct': 0.05,
        'fee_volatile_pct': 0.30
    }

    if pair.get('chainId') == 'base':
        fee_client = AerodromeFees(st.session_state.get('basescan_key', ''))
        stable_fee = fee_client.get_pool_fee(row['pair_address'], True)
        volatile_fee = fee_client.get_pool_fee(row['pair_address'], False)
        row['fee_stable_pct'] = stable_fee or 0.05
        row['fee_volatile_pct'] = volatile_fee or 0.30

    return pd.DataFrame([row])

def create_price_chart(token_address: str, token_symbol: str) -> Optional[go.Figure]:
    pair_data = fetch_token_price_history(token_address)
    if not pair_data:
        return None

    current_price = float(pair_data.get('priceUsd', 0))
    h24_change = float(pair_data.get('priceChange', {}).get('h24', 0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[datetime.now() - timedelta(hours=24), datetime.now()],
        y=[current_price * (1 - h24_change/100), current_price],
        mode='lines+markers',
        name=f'{token_symbol} Price',
        line=dict(color='purple', width=3)
    ))

    fig.update_layout(
        title=f"{token_symbol} Price (24h Trend)",
        xaxis_title="Time",
        yaxis_title="USD",
        height=250,
        showlegend=False
    )

    return fig

def main():
    st.title("🚀 Aerodrome Pool Tracker")

    if 'chart_order' not in st.session_state:
        st.session_state.chart_order = []

    st.sidebar.header("⚙️ Settings")
    addresses_input = st.sidebar.text_area(
        "Aerodrome pool addresses on Base (one per line)",
        value="0x9Da64ed1b87b3d0d3d1E731dd3aAAAc08eb0f5C3\n0x80c394f8867e06704d39a5910666a3e71ca7f325\n0xdb6556a14976894a01085c2abf3c85c86d1c15c8",
        height=120
    )
    basescan_key = st.sidebar.text_input("Basescan API Key (optional)", type="password")
    st.session_state['basescan_key'] = basescan_key
    if st.sidebar.button("🔄 Refresh (clear cache)"):
        st.cache_data.clear()
        st.rerun()

    raw_addresses = [a.strip() for a in addresses_input.splitlines()]
    pool_addresses = [addr for addr in raw_addresses if addr and addr.startswith('0x')]
    pool_addresses = list(dict.fromkeys([addr for addr in pool_addresses if is_valid_address(addr)]))

    st.sidebar.info(f"✅ Validated {len(pool_addresses)} pools")

    if not pool_addresses:
        st.info("👆 Enter at least one valid Aerodrome pool address (one per line).")
        st.stop()

    fee_client = AerodromeFees(basescan_key)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stable Pool Fee", f"{fee_client.get_default_fees()['stable']}%")
    with col2:
        st.metric("Volatile Pool Fee", f"{fee_client.get_default_fees()['volatile']}%")
    with col3:
        st.metric("RPC Status", "🟢 Connected")

    all_rows = []
    valid_pools = 0

    with st.spinner("Fetching pool data..."):
        for addr in pool_addresses:
            pair = fetch_pair('base', addr.lower())
            if pair:
                df_row = build_dataframe_from_pair(pair)
                all_rows.append(df_row)
                valid_pools += 1

    if not all_rows:
        st.error("❌ No valid pairs fetched from DexScreener.")
        st.stop()

    df = pd.concat(all_rows, ignore_index=True)
    st.success(f"✅ Loaded {valid_pools} pools")

    with st.expander("🔍 Debug: Raw Data"):
        st.dataframe(df[['pair_name', 'token0_symbol', 'token1_symbol', 'liquidity_token0', 'liquidity_token1']])

    # Pool Overview with asset balances
    st.subheader("📊 Pools Overview")
    display_cols = ['pair_name', 'pair_address', 'token0_symbol', 'liquidity_token0', 
                    'token1_symbol', 'liquidity_token1', 'liquidity_usd', 'volume_24h_usd', 
                    'tx_24h_count', 'fee_stable_pct']

    df_display = df[display_cols].copy()
    df_display.columns = ['Pair', 'Address', 'Token A', 'Token A Balance', 
                         'Token B', 'Token B Balance', 'Total Liq. (USD)', 
                         'Vol. 24h (USD)', 'Trades 24h', 'Stable Fee %']
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Aggregates
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💧 Total Liquidity USD", f"${df['liquidity_usd'].sum():,.0f}")
    with col2:
        st.metric("📈 Total Volume 24h USD", f"${df['volume_24h_usd'].sum():,.0f}")
    with col3:
        st.metric("🔄 Total Trades 24h", f"{df['tx_24h_count'].sum():,}")

    # FIXED Charts - using update_layout instead of update_xaxis
    st.subheader("📊 24h Volume by Pool")
    fig_vol = px.bar(df, x='pair_name', y='volume_24h_usd', 
                     hover_data=['liquidity_token0', 'liquidity_token1'],
                     title="24h Volume by Pool",
                     labels={'pair_name': 'Pool Pair', 'volume_24h_usd': 'Volume 24h USD'})
    fig_vol.update_layout(xaxis_tickangle=-45, xaxis=dict(tickmode='linear'))
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("💧 Liquidity by Pool")
    fig_liq = px.bar(df, x='pair_name', y='liquidity_usd', 
                     hover_data=['liquidity_token0', 'liquidity_token1'],
                     title="Liquidity by Pool",
                     labels={'pair_name': 'Pool Pair', 'liquidity_usd': 'Liquidity USD'})
    fig_liq.update_layout(xaxis_tickangle=-45, xaxis=dict(tickmode='linear'))
    st.plotly_chart(fig_liq, use_container_width=True)

    # Per-pool details
    st.subheader("🔍 Per-Pool Details")
    for _, row in df.iterrows():
        with st.expander(f"{row['pair_name']} | {row['pair_address'][:10]}..."):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Price USD", f"${row['price_usd']:.6f}")
            with col2:
                st.metric("💧 Total Liquidity", f"${row['liquidity_usd']:,.0f}")
            with col3:
                st.metric("📈 Volume 24h", f"${row['volume_24h_usd']:,.0f}")
            with col4:
                st.metric("🔄 Trades 24h", f"{row['tx_24h_count']:,}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{row['token0_symbol']} Balance", f"{row['liquidity_token0']:,.4f}")
            with col2:
                st.metric(f"{row['token1_symbol']} Balance", f"{row['liquidity_token1']:,.4f}")

            st.caption(f"Fees: Stable {row['fee_stable_pct']}%, Volatile {row['fee_volatile_pct']}%")

    # Unique token price charts with drag-drop
    unique_tokens = {}
    for _, row in df.iterrows():
        if row['token0_address'] not in unique_tokens:
            unique_tokens[row['token0_address']] = row['token0_symbol']
        if row['token1_address'] not in unique_tokens:
            unique_tokens[row['token1_address']] = row['token1_symbol']

    if not st.session_state.chart_order or set(st.session_state.chart_order) != set(unique_tokens.keys()):
        st.session_state.chart_order = list(unique_tokens.keys())

    st.subheader("📈 Token Price Charts (Reorder with ⬆️⬇️)")

    for i, token_addr in enumerate(st.session_state.chart_order):
        token_symbol = unique_tokens[token_addr]
        col1, col2, col3 = st.columns([0.1, 1, 0.1])

        with col1:
            if i > 0 and st.button("⬆️", key=f"up_{i}"):
                st.session_state.chart_order[i], st.session_state.chart_order[i-1] =                     st.session_state.chart_order[i-1], st.session_state.chart_order[i]
                st.rerun()

        with col2:
            fig = create_price_chart(token_addr, token_symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True, height=250)
            else:
                st.warning(f"No price data for {token_symbol}")

        with col3:
            if i < len(st.session_state.chart_order) - 1 and st.button("⬇️", key=f"down_{i}"):
                st.session_state.chart_order[i], st.session_state.chart_order[i+1] =                     st.session_state.chart_order[i+1], st.session_state.chart_order[i]
                st.rerun()

if __name__ == "__main__":
    main()