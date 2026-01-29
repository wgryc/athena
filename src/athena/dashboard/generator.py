from datetime import datetime
from decimal import Decimal
from pathlib import Path
import json

from jinja2 import Environment, FileSystemLoader

from ..currency import Currency
from ..portfolio import Portfolio, calculate_portfolio_value_by_day
from ..metrics import (
    calculate_daily_returns,
    calculate_sharpe_ratio_cumulative,
    calculate_sharpe_ratio_by_day_cumulative,
)


TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_dashboard(
    portfolio: Portfolio,
    target_currency: Currency,
    annual_risk_free_rate: float,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    title: str = "Portfolio Dashboard",
    periods_in_year: int = 365,
) -> str:
    """
    Generate an HTML dashboard for a portfolio.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to display values in.
        annual_risk_free_rate: Annual nominal risk-free rate for Sharpe calculation.
        start_date: Start date for the analysis period.
        end_date: End date for the analysis period.
        title: Title to display on the dashboard.
        periods_in_year: Trading periods in a year (365 for daily).

    Returns:
        HTML string of the complete dashboard.
    """
    # Get portfolio values over time
    portfolio_values = calculate_portfolio_value_by_day(
        portfolio,
        target_currency,
        start_date,
        end_date
    )

    if not portfolio_values:
        raise ValueError("No portfolio values available for the given date range.")

    sorted_dates = sorted(portfolio_values.keys())

    # Calculate metrics
    start_value = portfolio_values[sorted_dates[0]]
    end_value = portfolio_values[sorted_dates[-1]]
    total_return = float((end_value - start_value) / start_value * 100) if start_value != 0 else 0

    # Calculate Sharpe ratio
    try:
        daily_sharpe, annual_sharpe = calculate_sharpe_ratio_cumulative(
            portfolio,
            target_currency,
            annual_risk_free_rate,
            start_date,
            end_date,
            periods_in_year
        )
    except ValueError:
        daily_sharpe, annual_sharpe = None, None

    # Calculate daily returns for chart
    daily_returns = calculate_daily_returns(portfolio_values)

    # Calculate cumulative Sharpe over time for chart
    sharpe_by_day = calculate_sharpe_ratio_by_day_cumulative(
        portfolio,
        target_currency,
        annual_risk_free_rate,
        start_date,
        end_date,
        periods_in_year
    )

    # Prepare chart data
    value_chart_data = {
        "labels": [d.strftime("%Y-%m-%d") for d in sorted_dates],
        "values": [float(portfolio_values[d]) for d in sorted_dates]
    }

    returns_chart_data = {
        "labels": [d.strftime("%Y-%m-%d") for d in sorted(daily_returns.keys())],
        "values": [daily_returns[d] * 100 for d in sorted(daily_returns.keys())]  # Convert to percentage
    }

    sharpe_dates = sorted(sharpe_by_day.keys())
    sharpe_chart_data = {
        "labels": [d.strftime("%Y-%m-%d") for d in sharpe_dates],
        "values": [sharpe_by_day[d][1] for d in sharpe_dates]  # Annual Sharpe
    }

    # Prepare template context
    context = {
        "title": title,
        "currency": target_currency.value,
        "start_date": sorted_dates[0].strftime("%Y-%m-%d"),
        "end_date": sorted_dates[-1].strftime("%Y-%m-%d"),
        "start_value": float(start_value),
        "end_value": float(end_value),
        "total_return": total_return,
        "daily_sharpe": daily_sharpe,
        "annual_sharpe": annual_sharpe,
        "risk_free_rate": annual_risk_free_rate * 100,
        # Formatted values for display
        "start_value_formatted": f"{float(start_value):,.2f}",
        "end_value_formatted": f"{float(end_value):,.2f}",
        "total_return_formatted": f"{total_return:.2f}",
        "risk_free_rate_formatted": f"{annual_risk_free_rate * 100:.2f}",
        "annual_sharpe_formatted": f"{annual_sharpe:.2f}" if annual_sharpe is not None else "N/A",
        # Chart data
        "value_chart_data": json.dumps(value_chart_data),
        "returns_chart_data": json.dumps(returns_chart_data),
        "sharpe_chart_data": json.dumps(sharpe_chart_data),
    }

    # Render template
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    template = env.get_template("dashboard.html")

    return template.render(**context)
