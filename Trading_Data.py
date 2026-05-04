import yfinance as yf
import pandas as pd
import pandas_ta as ta


def get_stock_data(ticker: str) -> pd.DataFrame | None:
    df = yf.download(ticker, period="20y", interval="1d", auto_adjust=True, progress=False)

    if df.empty:
        return None

    # yfinance >= 0.2.x returns MultiIndex columns (like  ('NVDIA', 'AAPL')) — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]

    df["RSI"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"])
    if macd is not None:
        df = pd.concat([df, macd], axis=1)

    return df
