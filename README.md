# Agentic Toolkit for Holistic Economic Narratives and Analysis (ATHENA, v0.1.0)

Track your portfolio, individual positions, forex, and other financial instruments in one place. Pull data from the Federal Reserve, Yahoo! Finance, and custom sources. Track metrics to benchmark yourself against hedge funds. Use LLMs to manage your portfolio.

Athena is a toolkit designed to easily track investments and delegate decision-making to a large language model. Our vision is to have LLMs help with tracking major news developments, run scenario analyses, and ultimately open and close positions.

The data structure lives in Excel; it can be edited by you to input positions, override choices, or incorporate your existing portfolios.

**This is an experimental toolkit.** It should be used to inform your research and does not represent any sort of official investment advice.

## Installation

Install via GitHub:

```bash
pip install git+https://github.com/wgryc/athena.git
```

If you are cloning the repo and installing from local source, run the following: `pip install -e .` or `pip install -e ".[dev]"` to run tests.

## Initial Runs and Reports

Track your portfolio positions, currencies, and values:

```bash
athena report sample_transactions.xlsx
```

Hedge fund metrics around your portfolio performance:

```bash
athena metrics sample_transactions.xlsx
```

## Data Structure

Transactions and portfolio information are stored in Excel files and are easily editable by users. A transactions file needs the following columns:

- SYMBOL: the ticker symbol. By default, use the Yahoo Finance! ticker format. However, you can use anything that is supported by the pricing APIs you are using.
- DATE AND TIME: ISO format for the date-time transaction is preferred, but you can also use standard Excel dates. If time zone information is missing, we assume NYC time. If times are missing (i.e., only dates are provided) we assume 12pm NYC time.
- TRANSACTION TYPE: we support BUY, SELL, CASH_IN, CASH_OUT, DIVIDEND, INTEREST, FEE, and CURRENCY_EXCHANGE.
- PRICE: the price paid per unit of SYMBOL.
- QUANTITY: the quantity being purchased.
- CURRENCY: the currency being used. We currently support USD, CAD, EUR, TWD, SGD, AUD, JPY, KRW, GBP, BRL, CNY, HKD, MXN, ZAR, CHF, and THB.

When logging currency exchanges, the SYMBOL is the target currency (e.g. "HKD") and the CURRENCY is the source currency (e.g., "USD"). The price is how much of the CURRENCY it costs to buy 1 unit of the target (SYMBOL) currency.

## Agentic Trading

Agents require connections to third-party APIs and API keys should be stored in a local `.env` file. If you are trading stocks, we recommend using the [Massive](https://massive.com/) API while for CBOE-traded commodities, we recommend [DataBento](https://databento.com/). Finally, the current agents use the Emerging Trajectories "events" API to get information and trade on it.

```bash
athena demo --commodities demo_commodities.xlsx
```

```bash
athena demo --meme-stocks demo_meme_stocks.xlsx
```

```bash
athena demo --us-stocks demo_us_stocks.xlsx
```

## Dashboards

Generate interactive HTML dashboards to visualize your portfolio performance, returns, and Sharpe ratio over time.

```bash
athena dashboard sample_transactions.xlsx
```

This creates a `portfolio_dashboard.html` file in the current directory. To specify a custom output filename:

```bash
athena dashboard sample_transactions.xlsx --output my_dashboard.html
```

If the output file already exists, Athena will automatically append "copy" to the filename to avoid overwriting.
