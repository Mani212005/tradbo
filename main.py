
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_backtest(symbol, start_date, end_date, sma_short=5, sma_long=20, rsi_period=14, rsi_overbought=70):
    print(f"--- Running backtest for {symbol} ---")

    # 1. Fetch historical data
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        print(f"Could not fetch data for {symbol}. Skipping.")
        return

    # 2. Calculate indicators
    data['SMA_Short'] = data['Close'].rolling(window=sma_short).mean()
    data['SMA_Long'] = data['Close'].rolling(window=sma_long).mean()
    data['RSI'] = compute_rsi(data['Close'], rsi_period)

    # 3. Generate signals
    data['Signal'] = 0
    buy_dates = []
    sell_dates = []
    position = 0  # 1 if holding, 0 if not
    trade_entries = [] # To store buy prices for win rate calculation

    for i in range(1, len(data)):
        # Buy condition
        if (
            data['SMA_Short'].iloc[i-1] < data['SMA_Long'].iloc[i-1]
            and data['SMA_Short'].iloc[i] > data['SMA_Long'].iloc[i]
            and data['RSI'].iloc[i] < rsi_overbought
            and position == 0
        ):
            data.at[data.index[i], 'Signal'] = 1
            buy_dates.append(data.index[i])
            trade_entries.append(data['Close'].iloc[i])
            position = 1
        # Sell condition
        elif (
            (
                data['SMA_Short'].iloc[i-1] > data['SMA_Long'].iloc[i-1]
                and data['SMA_Short'].iloc[i] < data['SMA_Long'].iloc[i]
            )
            or (data['RSI'].iloc[i] > rsi_overbought)
        ) and position == 1:
            data.at[data.index[i], 'Signal'] = -1
            sell_dates.append(data.index[i])
            if trade_entries: # Ensure there was a buy to match with this sell
                buy_price = trade_entries.pop(0) # Get the price of the oldest open trade
                sell_price = data['Close'].iloc[i]
                # Log the trade outcome for win rate calculation
                # For simplicity, assuming one share per trade for win rate
                # In a real scenario, you'd track actual shares and profit/loss
                if sell_price > buy_price:
                    data.at[data.index[i], 'Trade_Outcome'] = 'Win'
                else:
                    data.at[data.index[i], 'Trade_Outcome'] = 'Loss'
            position = 0

    # 4. Simulate trades and calculate returns
    capital = 100000  # initial capital
    shares = 0
    cash = capital
    trade_log = []
    
    # Initialize equity curve with initial capital
    equity_curve = [capital] * len(data)

    for i, row in data.iterrows():
        current_value = cash + shares * row['Close']
        equity_curve[data.index.get_loc(i)] = current_value # Update equity for current day

        if row['Signal'] == 1 and cash > 0:
            shares_to_buy = cash // row['Close']
            if shares_to_buy > 0:
                shares += shares_to_buy
                cash -= shares_to_buy * row['Close']
                trade_log.append({'Date': i, 'Type': 'Buy', 'Price': row['Close'], 'Shares': shares_to_buy})
        elif row['Signal'] == -1 and shares > 0:
            cash += shares * row['Close']
            trade_log.append({'Date': i, 'Type': 'Sell', 'Price': row['Close'], 'Shares': shares})
            shares = 0
    
    # Final value if holding at the end of the period
    final_value = cash + shares * data['Close'].iloc[-1]
    strategy_return = (final_value - capital) / capital * 100

    # Market return (Buy and Hold)
    market_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100

    # Calculate Win Rate
    wins = data[data['Trade_Outcome'] == 'Win'].shape[0]
    losses = data[data['Trade_Outcome'] == 'Loss'].shape[0]
    total_completed_trades = wins + losses
    win_rate = (wins / total_completed_trades * 100) if total_completed_trades > 0 else 0

    # 5. Plotting
    # Price with SMA Crossovers and Buy/Sell Signals
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', alpha=0.7)
    plt.plot(data['SMA_Short'], label=f'SMA {sma_short}', alpha=0.7)
    plt.plot(data['SMA_Long'], label=f'SMA {sma_long}', alpha=0.7)
    
    # Plotting signals as markers on the close price line
    plt.scatter(buy_dates, data.loc[buy_dates]['Close'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)
    plt.scatter(sell_dates, data.loc[sell_dates]['Close'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)
    
    plt.title(f'{symbol} Price with SMA Crossovers and Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Equity curve
    plt.figure(figsize=(12, 5))
    plt.plot(data.index, equity_curve, label='Strategy Equity Curve')
    plt.plot(data.index, capital * (data['Close'] / data['Close'].iloc[0]), label='Market (Buy & Hold)')
    plt.title(f'Equity Curve: Strategy vs Market for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 6. Print summary
    print(f"
--- Performance Summary for {symbol} ---")
    print(f"Strategy Return: {strategy_return:.2f}%")
    print(f"Market Return:   {market_return:.2f}%")
    print(f"Total Trades:    {len(trade_log)}")
    print(f"Win Rate:        {win_rate:.2f}%")
    if trade_log:
        buys = [t for t in trade_log if t['Type'] == 'Buy']
        sells = [t for t in trade_log if t['Type'] == 'Sell']
        print(f"Total Buys:      {len(buys)}")
        print(f"Total Sells:     {len(sells)}")
    else:
        print("No trades executed.")

if __name__ == '__main__':
    # Define parameters for the backtest
    STOCKS = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS']
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'

    for stock_symbol in STOCKS:
        run_backtest(stock_symbol, START_DATE, END_DATE)
 