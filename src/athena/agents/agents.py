from dotenv import load_dotenv
import os
import time
from pathlib import Path
from anthropic import Anthropic
import litellm
from massive import RESTClient
import json
from jinja2 import Environment, FileSystemLoader

import requests

from decimal import Decimal

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Set up Jinja2 template environment
TEMPLATES_DIR = Path(__file__).parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

from ..portfolio import (
    Portfolio,
    load_portfolio_from_excel,
    save_portfolio_to_excel,
    TransactionType,
    get_positions,
    get_cash_balances,
    calculate_portfolio_value_on_date,
)
from ..currency import Currency

SLEEP_TIMES_FOR_ANTHROPIC = [60, 120, 300]  # in seconds

load_dotenv()
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
#DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-5"

ET_API_KEY = os.getenv("ET_API_KEY", "")
REQUEST_URL = "https://v2.emergingtrajectories.com/p/api/v0/get_events"

import databento
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")

# Dataset constants
COMMODITY_DATASET_CME = "GLBX.MDP3"  # CME Globex - futures & commodities (includes NYMEX)

# Commodities we're tracking (all on CME/NYMEX via GLBX.MDP3)
# year_digits: 1 = single digit (CLH6), 2 = two digits (NGG26)
COMMODITY_INFO = {
    "NG": {"name": "Natural Gas", "year_digits": 2},
    "CL": {"name": "WTI Crude Oil", "year_digits": 1},
    "RB": {"name": "RBOB Gasoline", "year_digits": 1},
}

# For backward compatibility
COMMODITY_ROOTS = {k: v["name"] for k, v in COMMODITY_INFO.items()}

# Month codes for futures
COMMODITY_MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
}

def commodities_get_quote(symbol: str, dataset: str = COMMODITY_DATASET_CME) -> tuple[float, float]:
    """
    Get the latest price for a futures symbol from Databento.
    Returns (ask_price, bid_price) from the most recent top-of-book data.

    Args:
        symbol: Futures symbol (e.g., "NGH26" for Natural Gas Mar 2026, "CLH6" for WTI Mar 2026)
        dataset: Databento dataset ID (default: GLBX.MDP3 for CME/NYMEX)

    Returns (-1.0, -1.0) on error.
    """
    # Try multiple symbol formats since Databento may use different conventions
    symbol_variants = [
        symbol,  # Original: CLG26
        f"{symbol}.FUT",  # With suffix: CLG26.FUT
        symbol.upper(),  # Ensure uppercase
    ]

    for sym in symbol_variants:
        try:
            client = databento.Historical(DATABENTO_API_KEY)

            # Get data from the last few days to find the most recent quote
            # Note: Historical data has ~20min delay, so end 30min ago to stay in range
            end = datetime.now(timezone.utc) - timedelta(minutes=30)
            start = end - timedelta(days=3)

            # Use tbbo (top of book) schema for bid/ask data
            data = client.timeseries.get_range(
                dataset=dataset,
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

            # tbbo schema has bid_px_00 and ask_px_00 columns (level 0)
            bid_price = float(latest["bid_px_00"])
            ask_price = float(latest["ask_px_00"])

            if sym != symbol:
                print(f"Note: Found data using symbol variant '{sym}' instead of '{symbol}'")

            return ask_price, bid_price

        except Exception:
            continue  # Try next variant

    print(f"No data found for {symbol} in dataset {dataset} (tried variants: {symbol_variants})")
    return -1.0, -1.0


def commodities_get_quote_for_context(symbol: str) -> str:
    """
    Get comprehensive quote information for a futures symbol to provide context to an LLM.
    Includes bid/ask, recent price data, and price movements.
    """
    result_parts = []

    # Identify the commodity type
    root = symbol[:2]
    commodity_name = COMMODITY_ROOTS.get(root, "Unknown Commodity")
    result_parts.append(f"Futures Contract: {symbol} ({commodity_name})")

    # 1. Get current bid/ask
    ask_price, bid_price = commodities_get_quote(symbol)
    if ask_price == -1.0 and bid_price == -1.0:
        result_parts.append("Bid/Ask: Unable to fetch")
        return "\n".join(result_parts)

    result_parts.append(f"Current Bid: ${bid_price:.4f}, Ask: ${ask_price:.4f}, Spread: ${(ask_price - bid_price):.4f}")

    # 2. Check futures market hours (CME Globex: Sunday 6pm - Friday 5pm ET, with daily break 5pm-6pm ET)
    et_tz = ZoneInfo('America/New_York')
    now_et = datetime.now(et_tz)
    day_of_week = now_et.weekday()  # 0=Monday, 6=Sunday
    hour = now_et.hour

    # Market is closed: Saturday all day, Sunday before 6pm, Friday after 5pm, and daily 5pm-6pm
    is_daily_break = 17 <= hour < 18
    is_weekend_closed = (day_of_week == 5) or (day_of_week == 6 and hour < 18) or (day_of_week == 4 and hour >= 17)
    market_is_open = not is_weekend_closed and not is_daily_break

    if market_is_open:
        result_parts.append("Market Status: Open (CME Globex)")
    else:
        result_parts.append("Market Status: Closed")

    # 3. Get historical price data for context using tbbo schema (more reliable)
    try:
        client = databento.Historical(DATABENTO_API_KEY)
        # End yesterday to avoid schema availability issues with current day
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        one_week_ago = end - timedelta(days=7)

        # Use the correct dataset for this symbol
        dataset = COMMODITY_DATASET_CME

        # Use tbbo schema since ohlcv-1d may not be available for recent dates
        data = client.timeseries.get_range(
            dataset=dataset,
            symbols=symbol,
            schema="tbbo",
            start=one_week_ago.isoformat(),
            end=end.isoformat(),
        )

        df = data.to_df()

        if not df.empty and len(df) >= 2:
            # Get prices from tbbo data (use ask_px_00 as reference price)
            current_price = float(df.iloc[-1]["ask_px_00"])
            week_ago_price = float(df.iloc[0]["ask_px_00"])
            week_change = current_price - week_ago_price
            week_pct = (week_change / week_ago_price) * 100
            result_parts.append(f"Week Ago Price: ${week_ago_price:.4f}")
            result_parts.append(f"1-Week Change: {week_change:+.4f} ({week_pct:+.2f}%)")

    except Exception:
        # Historical data is optional, don't clutter output with errors
        pass

    return "\n".join(result_parts)


def commodities_format_year(year: int, year_digits: int) -> str:
    """Format year based on commodity convention (1 or 2 digits)."""
    if year_digits == 1:
        return str(year % 10)
    else:
        return f"{year % 100:02d}"


def commodities_get_front_month_symbols() -> list[str]:
    """
    Generate front-month and next-month symbols for our tracked commodities.

    Note: Different commodities use different year formats:
    - NG (Natural Gas): 2-digit years (NGG26)
    - CL (WTI Crude): 1-digit years (CLH6)
    """
    now = datetime.now(timezone.utc)
    current_month = now.month
    current_year = now.year

    symbols = []
    for root, info in COMMODITY_INFO.items():
        year_digits = info["year_digits"]

        # Front month (current month if early, otherwise next month)
        if now.day > 20:  # After 20th, use next month as "front"
            front_month = current_month + 1
            front_year = current_year
            if front_month > 12:
                front_month = 1
                front_year += 1
        else:
            front_month = current_month
            front_year = current_year

        # Second month
        second_month = front_month + 1
        second_year = front_year
        if second_month > 12:
            second_month = 1
            second_year += 1

        # Third month
        third_month = second_month + 1
        third_year = second_year
        if third_month > 12:
            third_month = 1
            third_year += 1

        symbols.append(f"{root}{COMMODITY_MONTH_CODES[front_month]}{commodities_format_year(front_year, year_digits)}")
        symbols.append(f"{root}{COMMODITY_MONTH_CODES[second_month]}{commodities_format_year(second_year, year_digits)}")
        symbols.append(f"{root}{COMMODITY_MONTH_CODES[third_month]}{commodities_format_year(third_year, year_digits)}")

    return symbols

def get_system_prompt(data_source_description: str, data_summary_description: str, system_prompt_file:str = "system_prompt.j2",) -> str:
    """Generate system prompt from template with the given descriptions."""
    template = jinja_env.get_template(system_prompt_file)
    return template.render(
        data_source_description=data_source_description,
        data_summary_description=data_summary_description,
    )


def get_email_summary_instructions() -> str:
    """Load email summary instructions from template."""
    template = jinja_env.get_template("email_summary.j2")
    return template.render()

class EmergingTrajectoriesEventsProxy:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.request_url = REQUEST_URL

    def get_events(self, project_code_csv:str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "project_codes": project_code_csv,
            "hours": 36
        }

        response = requests.post(self.request_url, headers=headers, json=payload)

        json_data = response.json()
        return json_data['data']

def get_quote(symbol: str) -> tuple[float, float]:
    try:
        client = RESTClient(MASSIVE_API_KEY)
        quote = client.get_last_quote(symbol)
        return quote.ask_price, quote.bid_price # type: ignore
    except Exception as e:
        print(f"Error fetching quote for {symbol}: {e}")
        return -1.0, -1.0

def get_quote_for_context(symbol: str) -> str:
    """
    Get comprehensive quote information for a symbol to provide context to an LLM.
    Includes bid/ask, current day OHLC (if market open), and price movements.
    """
    client = RESTClient(MASSIVE_API_KEY)
    result_parts = []
    result_parts.append(f"Stock: {symbol}")

    # 1. Get current bid/ask
    try:
        quote = client.get_last_quote(symbol)
        bid_price = quote.bid_price
        ask_price = quote.ask_price
        result_parts.append(f"Current Bid: ${bid_price:.2f}, Ask: ${ask_price:.2f}, Spread: ${(ask_price - bid_price):.2f}")
    except Exception as e:
        result_parts.append(f"Bid/Ask: Unable to fetch ({e})")
        return "\n".join(result_parts)

    # 2. Check if market is currently open and get today's OHLC
    et_tz = ZoneInfo('America/New_York')
    now_et = datetime.now(et_tz)
    market_open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    is_weekday = now_et.weekday() < 5  # Monday = 0, Friday = 4
    is_market_hours = market_open_time <= now_et <= market_close_time
    market_is_open = is_weekday and is_market_hours

    today_str = now_et.strftime("%Y-%m-%d")
    current_price = None

    if market_is_open:
        try:
            aggs = list(client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=today_str,
                to=today_str,
                limit=1
            ))
            if aggs:
                today_agg = aggs[0]
                current_price = today_agg.close
                result_parts.append(f"Today (Market Open): Price: ${today_agg.close:.2f}, Open: ${today_agg.open:.2f}, High: ${today_agg.high:.2f}, Low: ${today_agg.low:.2f}")
        except Exception as e:
            result_parts.append(f"Today's OHLC: Unable to fetch ({e})")
    else:
        result_parts.append("Market Status: Closed")

    # 3-5. Get price movements for 1 week, 1 month, 1 year
    now_utc = datetime.now(timezone.utc)
    one_week_ago = (now_utc - timedelta(days=10)).strftime("%Y-%m-%d")  # Extra days to account for weekends
    one_month_ago = (now_utc - timedelta(days=35)).strftime("%Y-%m-%d")
    one_year_ago = (now_utc - timedelta(days=370)).strftime("%Y-%m-%d")

    try:
        # Get recent data to find current price if we don't have it
        recent_aggs = list(client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=one_week_ago,
            to=today_str,
            limit=50
        ))

        if recent_aggs:
            if current_price is None:
                current_price = recent_aggs[-1].close
                result_parts.append(f"Last Close: ${current_price:.2f}")

            # Week ago: find price from ~5-7 trading days ago
            week_ago_price = recent_aggs[0].close if len(recent_aggs) >= 5 else None

            # Get month ago price
            month_aggs = list(client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=one_month_ago,
                to=(now_utc - timedelta(days=25)).strftime("%Y-%m-%d"),
                limit=5
            ))
            month_ago_price = month_aggs[-1].close if month_aggs else None

            # Get year ago price
            year_aggs = list(client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=one_year_ago,
                to=(now_utc - timedelta(days=360)).strftime("%Y-%m-%d"),
                limit=5
            ))
            year_ago_price = year_aggs[-1].close if year_aggs else None

            # Build price movement summary
            movements = []
            if week_ago_price and current_price:
                week_change = current_price - week_ago_price
                week_pct = (week_change / week_ago_price) * 100
                movements.append(f"1-Week: {week_change:+.2f} ({week_pct:+.1f}%)")

            if month_ago_price and current_price:
                month_change = current_price - month_ago_price
                month_pct = (month_change / month_ago_price) * 100
                movements.append(f"1-Month: {month_change:+.2f} ({month_pct:+.1f}%)")

            if year_ago_price and current_price:
                year_change = current_price - year_ago_price
                year_pct = (year_change / year_ago_price) * 100
                movements.append(f"1-Year: {year_change:+.2f} ({year_pct:+.1f}%)")

            if movements:
                result_parts.append("Price Changes: " + ", ".join(movements))
    except Exception as e:
        result_parts.append(f"Price Movements: Unable to fetch ({e})")

    return "\n".join(result_parts)

def stream_llm_response(system_prompt:str, messages:list[dict[str, str]], model:str) -> str:
    max_num_attempts = 3
    num_attempts = 0

    # Format model name for litellm (prefix with anthropic/)
    litellm_model = f"anthropic/{model}"
    litellm_messages = [{"role": "system", "content": system_prompt}] + messages

    while num_attempts < max_num_attempts:
        try:
            response = litellm.completion(
                model=litellm_model,
                messages=litellm_messages,
                max_tokens=8192,
            )
            return response.choices[0].message.content  # type: ignore

        except Exception as e:
            print(f"Attempt {num_attempts} failed: {e}")
            print(f"Retrying after {SLEEP_TIMES_FOR_ANTHROPIC[num_attempts]} seconds...")
            time.sleep(SLEEP_TIMES_FOR_ANTHROPIC[num_attempts])
            num_attempts += 1

            if num_attempts >= max_num_attempts:
                raise e

    return ""  # Should never reach here

def format_positions_table(portfolio: Portfolio, current_time: datetime) -> str:
    """Format portfolio positions as a markdown table."""
    positions = get_positions(current_time, portfolio)
    if len(positions) == 0:
        return "No current positions."

    lines = [
        "| Symbol | Quantity | Book Value | Market Value | Gain/Loss % |",
        "|--------|----------|------------|--------------|-------------|",
    ]
    for pos in positions:
        gain_str = f"{pos.gain_loss_percent:+.2f}%" if pos.gain_loss_percent is not None else "N/A"
        book_str = f"${pos.book_value:,.2f}" if pos.book_value else "N/A"
        lines.append(f"| {pos.symbol} | {pos.quantity:.4f} | {book_str} | ${pos.total_value:,.2f} | {gain_str} |")
    return "\n".join(lines)


def format_cash_balances(portfolio: Portfolio, current_time: datetime) -> str:
    """Format cash balances as a string."""
    cash_balances = get_cash_balances(portfolio, current_time)
    balances_str = "\n".join(
        [f"{curr}: ${bal}" for curr, bal in cash_balances.items() if bal != Decimal("0")]
    )
    return balances_str if balances_str else "No cash balances."


def format_transaction_log(portfolio: Portfolio) -> str:
    """Format transaction log as a string."""
    lines = []
    for transaction in portfolio.transactions:
        lines.append(
            f"{transaction.transaction_datetime.isoformat()} - "
            f"{transaction.transaction_type.value} {transaction.quantity} shares of "
            f"{transaction.symbol} @ ${transaction.price}"
        )
    return "\n".join(lines)


def generate_initial_message(
    portfolio: Portfolio,
    project_code: str,
    events_section_title: str,
    events_description: str,
    use_commodities: bool = False,
) -> str:
    """Generate initial message from template with the given configuration."""
    current_time = datetime.now(timezone.utc)

    # Fetch events data
    etep = EmergingTrajectoriesEventsProxy(ET_API_KEY)
    events_data = etep.get_events(project_code)

    # Calculate portfolio data
    total_market_value = calculate_portfolio_value_on_date(portfolio, current_time, Currency.USD)

    # Generate available contracts section for commodities
    available_contracts_section = ""
    if use_commodities:
        front_months = commodities_get_front_month_symbols()
        available_contracts_section = """|--AVAILABLE CONTRACTS--|

You can trade the following futures contracts:
- Natural Gas (NG): NGF26, NGG26, NGH26, etc. (10,000 MMBtu per contract) - uses 2-digit year
- WTI Crude (CL): CLF6, CLG6, CLH6, etc. (1,000 barrels per contract) - uses 1-digit year
- RBOB Gasoline (RB): RBF6, RBG6, RBH6, etc. (42,000 gallons per contract) - uses 1-digit year

Symbol format: [ROOT][MONTH][YEAR]
Month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec

Suggested liquid contracts (front months): """ + ", ".join(front_months) + "\n"

    # Render template
    template = jinja_env.get_template("initial_message.j2")
    return template.render(
        current_datetime=current_time.isoformat(),
        events_section_title=events_section_title,
        events_description=events_description,
        events_data=events_data,
        positions_table=format_positions_table(portfolio, current_time),
        cash_balances=format_cash_balances(portfolio, current_time),
        total_market_value=total_market_value,
        transaction_log=format_transaction_log(portfolio),
        available_contracts_section=available_contracts_section,
    )

def run_investing_agent(
    portfolio_file_name: str,
    project_code: str,
    events_section_title: str,
    events_description: str,
    data_source_description: str,
    data_summary_description: str,
    generate_summary: bool = False,
    system_prompt_file:str = "system_prompt.j2",
    pricing_manager = None,
    use_commodities: bool = False
):
    """Run the investing agent with the given configuration.

    Args:
        portfolio_file_name: Path to the portfolio Excel file
        project_code: The project code for fetching events
        events_section_title: Title for the events section
        events_description: Description of the events data
        data_source_description: Description of the data source
        data_summary_description: Description for the data summary
        generate_summary: Whether to generate an email summary at the end
    """
    portfolio = load_portfolio_from_excel(
        portfolio_file_name,
        primary_currency=Currency.USD,
        create_if_missing=True,
        error_out_negative_cash=False,
        error_out_negative_quantity=True,
        pricing_manager=pricing_manager,
    )

    system_prompt = get_system_prompt(data_source_description, data_summary_description, system_prompt_file=system_prompt_file)
    initial_message = generate_initial_message(
        portfolio, project_code, events_section_title, events_description, use_commodities=use_commodities
    )

    messages_array = [
                        {
                            "role": "user",
                            "content": initial_message
                        }
                    ]

    func_get_quote = commodities_get_quote if use_commodities else get_quote
    func_get_quote_for_context = commodities_get_quote_for_context if use_commodities else get_quote_for_context

    while True:

        response = stream_llm_response(
            system_prompt=system_prompt,
            messages=messages_array,
            model=DEFAULT_ANTHROPIC_MODEL
        )

        messages_array.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        # print("Response from Anthropic:")
        # print(response)

        quotes_request = False
        new_content = ""

        if response.find("|--COMMAND--|") != -1:
            command_section = response.split("|--COMMAND--|")[1]
            command_section = command_section.strip()
            lines = command_section.split("\n")
            # print(lines)
            for line in lines:
                line = line.strip()
                if line.startswith("QUOTE:"):
                    quotes_request = True
                    line = line.replace("QUOTE:", "").strip()
                    symbols = line.split(",")
                    for symbol in symbols:
                        symbol = symbol.strip()
                        print(f"Fetching quote for symbol: {symbol}")
                        quote_for_context = func_get_quote_for_context(symbol)
                        new_content += f"\n{quote_for_context}\n"
                elif line.startswith("BUY:"):
                    line = line.replace("BUY:", "").strip()
                    parts = line.split(",")
                    for part in parts:
                        toks = part.strip().split("|")
                        if len(toks) == 2:
                            symbol = toks[0].strip()
                            quantity = toks[1].strip()
                            ask_price, bid_price = func_get_quote(symbol)
                            if ask_price == -1.0 and bid_price == -1.0:
                                print(f"Error fetching quote for {symbol}. Skipping BUY order.\n")
                                continue
                            print(f"Placing BUY order for {quantity} shares of {symbol}\n")
                            transaction = portfolio.add_transaction_now(
                                symbol=symbol,
                                transaction_type=TransactionType.BUY,
                                quantity=Decimal(quantity),
                                price=ask_price  # type: ignore[arg-type]
                            )
                        else:
                            print(f"Invalid BUY command format: {part}\n")
                elif line.startswith("SELL:"):
                    line = line.replace("SELL:", "").strip()
                    parts = line.split(",")
                    for part in parts:
                        toks = part.strip().split("|")
                        if len(toks) == 2:
                            symbol = toks[0].strip()
                            quantity = toks[1].strip()
                            ask_price, bid_price = func_get_quote(symbol)
                            if ask_price == -1.0 and bid_price == -1.0:
                                print(f"Error fetching quote for {symbol}. Skipping SELL order.\n")
                                continue
                            print(f"Placing SELL order for {quantity} shares of {symbol}\n")
                            transaction = portfolio.add_transaction_now(
                                symbol=symbol,
                                transaction_type=TransactionType.SELL,
                                quantity=Decimal(quantity),
                                price=bid_price  # type: ignore[arg-type]
                            )
                        else:
                            print(f"Invalid SELL command format: {part}\n")
                else:
                    print(f"Unknown command: {line}\n")

        if quotes_request:
            print("\n\n\nSENDING MESSAGE TO ANTHROPIC:\n")
            print(new_content)
            messages_array.append(
                {
                    "role": "user",
                    "content": new_content.strip()
                }
            )
        else:
            break 

    save_portfolio_to_excel(portfolio, portfolio_file_name)

    if generate_summary:
        email_instructions = get_email_summary_instructions()
        messages_array.append({
            "role": "user",
            "content": email_instructions
        })
        summary_response = stream_llm_response(
            system_prompt=email_instructions,
            messages=messages_array,
            model=DEFAULT_ANTHROPIC_MODEL
        )
        messages_array.append(
            {
                "role": "assistant",
                "content": summary_response
            }
        )

        print(json.dumps(messages_array, indent=4))

        return summary_response
    
    return ""