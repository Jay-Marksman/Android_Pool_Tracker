import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict

def get_timeframe_range(period: str) -> tuple[datetime, datetime]:
    """Compute start/end datetimes based on timeframe selection."""
    end = datetime.now()
    if period == "1H":
        start = end - timedelta(hours=1)
    elif period == "6H":
        start = end - timedelta(hours=6)
    elif period == "24H":
        start = end - timedelta(hours=24)
    elif period == "7D":
        start = end - timedelta(days=7)
    elif period == "30D":
        start = end - timedelta(days=30)
    else:  # "All"
        start = end - timedelta(days=365)  # Reasonable default
    return start, end

def filter_df_by_timeframe(df: pd.DataFrame, start: datetime, end: datetime, period: str) -> pd.DataFrame:
    """
    Filter or aggregate df for timeframe. Note: Current df has h1/h6/h24 fields;
    for longer periods like 7D/30D, assumes external historical fetch or approximation.
    Extend with historical API calls if needed.
    """
    # For demo with existing h1/h6/h24; scale for longer (placeholder logic)
    filtered = df.copy()
    if period == "1H":
        filtered['volume_usd'] = filtered['volume_1h_usd'].fillna(0)
        filtered['trades'] = filtered['tx_1h_count'].fillna(0)
    elif period == "6H":
        filtered['volume_usd'] = filtered['volume_6h_usd'].fillna(0)
        filtered['trades'] = filtered['tx_6h_count'].fillna(0)
    elif period == "24H":
        filtered['volume_usd'] = filtered['volume_24h_usd'].fillna(0)
        filtered['trades'] = filtered['tx_24h_count'].fillna(0)
    else:  # 7D/30D/All: approx as 24H * days (extend with real hist data)
        days = {'7D': 7, '30D': 30, 'All': 365}[period]
        filtered['volume_usd'] = filtered['volume_24h_usd'] * days
        filtered['trades'] = filtered['tx_24h_count'] * days
    filtered['token0_balance'] = filtered['liquidity_token0']
    filtered['token1_balance'] = filtered['liquidity_token1']
    filtered = filtered[filtered['volume_usd'] > 0]  # Hide zero-volume pools
    return filtered

def create_volume_chart(df_filtered: pd.DataFrame, period: str):
    fig = px.bar(df_filtered, x='pair_name', y='volume_usd', 
                 title=f'Volume by Pool ({period})',
                 labels={'volume_usd': 'Volume USD'})
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_trades_chart(df_filtered: pd.DataFrame, period: str):
    fig = px.bar(df_filtered, x='pair_name', y='trades', 
                 title=f'Trades by Pool ({period})',
                 labels={'trades': 'Trade Count'})
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def display_timeframe_analytics(df: pd.DataFrame, fee_client=None):
    """Main display function for timeframes section."""
    st.header("Pool Analytics by Timeframe")
    
    # Sidebar selector
    period = st.sidebar.selectbox("Select Timeframe", ["1H", "6H", "24H", "7D", "30D", "All"])
    start, end = get_timeframe_range(period)
    st.sidebar.info(f"Range: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")
    
    df_filtered = filter_df_by_timeframe(df, start, end, period)
    
    if df_filtered.empty:
        st.warning("No data for selected timeframe.")
        return
    
    # Aggregates
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Volume", f"${df_filtered['volume_usd'].sum():,.0f}")
    with col2:
        st.metric("Total Trades", f"{df_filtered['trades'].sum():,.0f}")
    with col3:
        total_liq = df_filtered['liquidity_usd'].sum()
        st.metric("Total Liquidity", f"${total_liq:,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_volume_chart(df_filtered, period), use_container_width=True)
    with col2:
        st.plotly_chart(create_trades_chart(df_filtered, period), use_container_width=True)
    
    # Balances table
    balances_df = df_filtered[['pair_name', 'token0_symbol', 'token0_balance', 
                               'token1_symbol', 'token1_balance']].copy()
    balances_df.columns = ['Pool', 'Token A', 'A Balance', 'Token B', 'B Balance']
    st.subheader("Token Balances")
    st.dataframe(balances_df, use_container_width=True, hide_index=True)
    
    # Optional: Fee preview (extend later)
    if fee_client:
        st.caption("Fees to be added in next iteration.")

if __name__ == "__main__":
    # Demo standalone
    st.title("PoolTimeframes Demo")
    demo_df = pd.DataFrame({
        'pair_name': ['Pool1', 'Pool2'], 'volume24h_usd': [10000, 5000],
        'tx24h_count': [50, 30], 'liquidity_token0': [1000, 500],
        'liquidity_token1': [2000, 1000], 'token0_symbol': ['A', 'B'],
        'token1_symbol': ['USDC', 'USDC'], 'liquidity_usd': [300000, 150000]
    })
    display_timeframe_analytics(demo_df)
