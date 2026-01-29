from abc import ABC, abstractmethod
from decimal import Decimal
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
import os
import sys

import yfinance as yf  # type: ignore[import-untyped]
import pandas as pd
from massive import RESTClient  # type: ignore[import-untyped]
import databento as db  # type: ignore[import-untyped]

from .currency import Currency

# Track which symbols have been force-refreshed this session
_refreshed_symbols: set[str] = set()


def get_yfinance_cache_path(symbol: str) -> Path:
    """Get the cache file path for a given symbol."""
    return Path.cwd() / ".cache" / "yfinance_prices" / f"{symbol}.csv"


def fetch_yfinance_data(symbol: str, min_date: date, max_date: date, force_cache_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch yfinance data for a symbol, using cache intelligently.

    If cached data exists, checks if it covers the requested date range.
    If not, expands the request to include all dates from cache + requested range,
    then updates the cache with the merged data.

    Args:
        symbol: The ticker symbol (e.g., "AAPL", "MSFT")
        min_date: The minimum date needed
        max_date: The maximum date needed

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    """
    cache_path = get_yfinance_cache_path(symbol)

    cached_df: pd.DataFrame | None = None
    cached_min: date | None = None
    cached_max: date | None = None

    # Always try to read cache first (needed for range calculation and fallback)
    if cache_path.exists():
        try:
            cached_df = pd.read_csv(cache_path, parse_dates=['Date'])  # type: ignore[call-overload]
            if not cached_df.empty:
                cached_df['Date'] = pd.to_datetime(cached_df['Date']).dt.date
                cached_min = cached_df['Date'].min()
                cached_max = cached_df['Date'].max()
        except Exception:
            # If cache is corrupted, we'll just refetch
            cached_df = None

    # Determine if we need to fetch new data
    fetch_start = min_date
    fetch_end = max_date

    # Check if we should force refresh (only once per symbol per session)
    should_force_refresh = force_cache_refresh and symbol not in _refreshed_symbols

    if should_force_refresh:
        # Force refresh: fetch fresh data but use cache for range expansion
        need_fetch = True
        if cached_df is not None and not cached_df.empty and cached_min is not None and cached_max is not None:
            # Expand fetch range to cover existing cache too, so we refresh everything
            fetch_start = min(min_date, cached_min)
            fetch_end = max(max_date, cached_max)
    elif cached_df is None or cached_df.empty or cached_min is None or cached_max is None:
        # No valid cache, fetch the full range
        need_fetch = True
    elif min_date < cached_min or max_date > cached_max:
        # We have cached data but it doesn't cover our range
        need_fetch = True
        fetch_start = min(min_date, cached_min)
        fetch_end = max(max_date, cached_max)
    else:
        # Cache covers our range
        need_fetch = False

    if need_fetch:
        # Track that we've refreshed this symbol
        _refreshed_symbols.add(symbol)

        # yfinance end date is exclusive, so add 1 day
        fetch_end_exclusive = fetch_end + timedelta(days=1)

        try:
            ticker = yf.Ticker(symbol)
            new_df: pd.DataFrame = ticker.history(  # type: ignore[call-arg]
                start=fetch_start.isoformat(),
                end=fetch_end_exclusive.isoformat(),
                auto_adjust=False
            )
        except Exception as e:
            # yfinance can fail if Yahoo Finance returns None (e.g., rate limiting,
            # requesting data for a date with no trading yet, network issues)
            print(f"Warning: yfinance request failed for {symbol}: {e}", file=sys.stderr)
            if cached_df is not None and not cached_df.empty:
                return cached_df
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

        if new_df.empty:
            # Empty response often indicates rate limiting from Yahoo Finance
            print(f"Warning: yfinance returned no data for {symbol} (possible rate limiting)", file=sys.stderr)
            # If we got no data but have cache, return cache
            if cached_df is not None and not cached_df.empty:
                return cached_df
            # Otherwise return empty DataFrame with expected columns
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

        # Reset index to make Date a column
        new_df = new_df.reset_index()
        new_df['Date'] = pd.to_datetime(new_df['Date']).dt.date

        # Keep only the columns we need
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_columns = [c for c in columns_to_keep if c in new_df.columns]
        new_df = new_df[available_columns]

        # Merge with cached data if we had any
        if cached_df is not None and not cached_df.empty:
            # Combine and remove duplicates, keeping newer data
            combined_df = pd.concat([cached_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        else:
            combined_df = new_df.sort_values('Date').reset_index(drop=True)

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(cache_path, index=False)

        return combined_df

    # Cache covers our range, return it (we know it's not None here)
    assert cached_df is not None
    return cached_df

class PricePoint:

    def __init__(self, symbol: str, price_datetime: datetime, price: Decimal, base_currency:Currency = Currency.USD):
        self.symbol: str = symbol
        self.price_datetime: datetime = price_datetime
        self.price: Decimal = price
        self.base_currency:Currency = base_currency

class PricingDataManager(ABC):
    
    @abstractmethod
    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        raise NotImplementedError("This method should be overridden by subclasses.")

# Returns the same price for any symbol/datetime.
class FixedPricingDataManager(PricingDataManager):
    
    def __init__(self, price_for_everything:Decimal = Decimal("1.0")):
        self.price = price_for_everything

    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:

        price_point = PricePoint(
            symbol=symbol,
            price_datetime=price_datetime,
            price=self.price,
            base_currency=Currency.USD
        )

        return price_point

class YFinancePricingDataManager(PricingDataManager):

    def __init__(self, force_cache_refresh: bool = False):
        self.force_cache_refresh = force_cache_refresh

    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        """Get price for a symbol on a specific date.

        If no data is available for the requested date (weekend, holiday),
        looks back up to 7 days to find the most recent trading day.

        For today's date during market hours, uses the current market price
        since the daily close isn't available yet.
        """
        target_date = price_datetime.date()
        today = date.today()

        # If requesting today's price, try to get the current/live market price first
        # since the daily close isn't available until market closes
        if target_date == today:
            try:
                ticker = yf.Ticker(symbol)
                # Use fast_info['lastPrice'] which provides the most recent price
                # including pre-market and after-hours trading, unlike regularMarketPrice
                # which only returns the last regular session close
                fast_info = ticker.fast_info
                current_price = fast_info.get('lastPrice')
                if current_price is not None:
                    price = Decimal(str(current_price)).quantize(Decimal("0.01"))
                    return PricePoint(
                        symbol=symbol,
                        price_datetime=datetime.now(),
                        price=price,
                        base_currency=Currency.USD
                    )
            except Exception:
                pass  # Fall through to historical data lookup

        # Fetch data - request a window to handle weekends/holidays
        # We ask for 10 days before to ensure we have data even if target is a Monday after a long weekend
        min_date = target_date - timedelta(days=10)
        max_date = target_date

        df = fetch_yfinance_data(symbol, min_date, max_date, self.force_cache_refresh)

        if df.empty:
            raise ValueError(f"No price data available for {symbol}")

        # Look for the target date or the most recent date before it
        for days_back in range(8):
            lookup_date = target_date - timedelta(days=days_back)
            matching_rows = df[df['Date'] == lookup_date]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                price = Decimal(str(row['Close'])).quantize(Decimal("0.01"))
                return PricePoint(
                    symbol=symbol,
                    price_datetime=datetime.combine(lookup_date, datetime.min.time()),
                    price=price,
                    base_currency=Currency.USD
                )

        raise ValueError(
            f"No price data available for {symbol} on {target_date} or the previous 7 days."
        )


class MassivePricingDataManager(PricingDataManager):
    """Pricing data manager using the Massive API."""

    def __init__(self, api_key: str):
        """Initialize the Massive client.

        Args:
            api_key: Massive API key.
        """
        self._client = RESTClient(api_key)

    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        """Get price for a symbol on a specific date.

        If no data is available for the requested date (weekend, holiday),
        looks back up to 7 days to find the most recent trading day.
        """
        target_date = price_datetime.date()

        # Request a window to handle weekends/holidays
        min_date = target_date - timedelta(days=10)
        max_date = target_date

        try:
            aggs = list(self._client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=min_date.isoformat(),
                to=max_date.isoformat(),
                limit=50
            ))
        except Exception as e:
            raise ValueError(f"Error fetching price data for {symbol}: {e}") from e

        if not aggs:
            raise ValueError(f"No price data available for {symbol}")

        # Convert aggs to a dict keyed by date for easy lookup
        price_by_date: dict[date, Decimal] = {}
        for agg in aggs:
            # Massive returns timestamp in milliseconds
            agg_date = datetime.fromtimestamp(agg.timestamp / 1000).date()  # type: ignore[union-attr]
            price_by_date[agg_date] = Decimal(str(agg.close)).quantize(Decimal("0.01"))  # type: ignore[union-attr]

        # Look for the target date or the most recent date before it
        for days_back in range(8):
            lookup_date = target_date - timedelta(days=days_back)
            if lookup_date in price_by_date:
                return PricePoint(
                    symbol=symbol,
                    price_datetime=datetime.combine(lookup_date, datetime.min.time()),
                    price=price_by_date[lookup_date],
                    base_currency=Currency.USD
                )

        raise ValueError(
            f"No price data available for {symbol} on {target_date} or the previous 7 days."
        )


class DatabentoPricingDataManager(PricingDataManager):
    """Pricing data manager using the Databento API for futures/commodities."""

    # Dataset constants
    DATASET_CME = "GLBX.MDP3"  # CME Globex - futures & commodities (includes NYMEX)

    def __init__(self, api_key: str | None = None, dataset: str = DATASET_CME):
        """Initialize the Databento client.

        Args:
            api_key: Databento API key. If None, reads from DATABENTO_API_KEY env var.
            dataset: Databento dataset ID (default: GLBX.MDP3 for CME/NYMEX futures)
        """
        self._api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self._api_key:
            raise ValueError("Databento API key required. Set DATABENTO_API_KEY env var or pass api_key.")
        self._dataset = dataset

    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        """Get price for a futures symbol on a specific date.

        Uses Databento's historical API to fetch top-of-book (tbbo) data.
        Returns the mid-price (average of bid and ask) for the most recent quote.

        If no data is available for the requested date, looks back up to 7 days
        to find the most recent trading day.

        Args:
            symbol: Futures symbol (e.g., "NGH26" for Natural Gas Mar 2026)
            price_datetime: The datetime to get the price for

        Returns:
            PricePoint with the mid-price from bid/ask data
        """
        target_date = price_datetime.date()

        # Try multiple symbol formats since Databento may use different conventions
        symbol_variants = [
            symbol,
            f"{symbol}.FUT",
            symbol.upper(),
        ]

        # Request a window to handle weekends/holidays
        # Databento historical data has ~20min delay, so end 30min ago for recent requests
        end_dt = datetime.now(timezone.utc) - timedelta(minutes=30)
        if price_datetime.date() < date.today():
            # For historical dates, use end of that day
            end_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)

        start_dt = datetime.combine(target_date - timedelta(days=10), datetime.min.time(), tzinfo=timezone.utc)

        for sym in symbol_variants:
            try:
                client = db.Historical(self._api_key)

                data = client.timeseries.get_range(
                    dataset=self._dataset,
                    symbols=sym,
                    schema="tbbo",
                    start=start_dt.isoformat(),
                    end=end_dt.isoformat(),
                )

                df = data.to_df()

                if df.empty:
                    continue  # Try next variant

                # Convert index to dates for grouping
                df['trade_date'] = pd.to_datetime(df.index).date

                # Look for the target date or the most recent date before it
                for days_back in range(8):
                    lookup_date = target_date - timedelta(days=days_back)
                    day_data = df[df['trade_date'] == lookup_date]

                    if not day_data.empty:
                        # Get the last quote of the day
                        latest = day_data.iloc[-1]
                        bid_price = float(latest["bid_px_00"])
                        ask_price = float(latest["ask_px_00"])

                        # Use mid-price (average of bid and ask)
                        mid_price = (bid_price + ask_price) / 2
                        price = Decimal(str(mid_price)).quantize(Decimal("0.0001"))

                        return PricePoint(
                            symbol=symbol,
                            price_datetime=datetime.combine(lookup_date, datetime.min.time()),
                            price=price,
                            base_currency=Currency.USD
                        )

            except Exception:
                continue  # Try next variant

        raise ValueError(
            f"No price data available for {symbol} on {target_date} or the previous 7 days. "
            f"Tried symbol variants: {symbol_variants}"
        )

    def get_quote(self, symbol: str) -> tuple[float, float]:
        """
        Get the latest bid/ask prices for a futures symbol.

        This method mirrors the interface from databentoagent.py for compatibility.

        Args:
            symbol: Futures symbol (e.g., "NGH26" for Natural Gas Mar 2026)

        Returns:
            Tuple of (ask_price, bid_price). Returns (-1.0, -1.0) on error.
        """
        symbol_variants = [
            symbol,
            f"{symbol}.FUT",
            symbol.upper(),
        ]

        for sym in symbol_variants:
            try:
                client = db.Historical(self._api_key)

                # Get data from the last few days to find the most recent quote
                # Historical data has ~20min delay, so end 30min ago
                end = datetime.now(timezone.utc) - timedelta(minutes=30)
                start = end - timedelta(days=3)

                data = client.timeseries.get_range(
                    dataset=self._dataset,
                    symbols=sym,
                    schema="tbbo",
                    start=start.isoformat(),
                    end=end.isoformat(),
                )

                df = data.to_df()

                if df.empty:
                    continue  # Try next variant

                # Get the most recent row
                latest = df.iloc[-1]

                bid_price = float(latest["bid_px_00"])
                ask_price = float(latest["ask_px_00"])

                return ask_price, bid_price

            except Exception:
                continue  # Try next variant

        return -1.0, -1.0