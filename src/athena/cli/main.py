#!/usr/bin/env python3
"""Main entry point for the ATHENA CLI."""

import argparse
import sys

ATHENA_LOGO = """
  █████╗ ████████╗██╗  ██╗███████╗███╗   ██╗ █████╗
 ██╔══██╗╚══██╔══╝██║  ██║██╔════╝████╗  ██║██╔══██╗
 ███████║   ██║   ███████║█████╗  ██╔██╗ ██║███████║
 ██╔══██║   ██║   ██╔══██║██╔══╝  ██║╚██╗██║██╔══██║
 ██║  ██║   ██║   ██║  ██║███████╗██║ ╚████║██║  ██║
 ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝
 ATHENA — Agentic Toolkit for Holistic Economic Narratives and Analysis
"""


def main():
    """Main entry point for the ATHENA CLI."""
    parser = argparse.ArgumentParser(
        prog="athena",
        description="ATHENA - Agentic Toolkit for Holistic Economic Narratives and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  athena report portfolio.xlsx              Display portfolio holdings report
  athena metrics portfolio.xlsx             Display portfolio metrics report
  athena report portfolio.xlsx -c CAD       Use CAD as primary currency
  athena metrics portfolio.xlsx --debug     Show debug info with prices
        """,
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands",
    )

    # Import subcommand modules and register them
    from .report import register_subcommand as register_report
    from .metrics import register_subcommand as register_metrics
    from .demo import register_subcommand as register_demo
    from .dashboard import register_subcommand as register_dashboard

    register_report(subparsers)
    register_metrics(subparsers)
    register_demo(subparsers)
    register_dashboard(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if args.command is None:
        print(ATHENA_LOGO)
        parser.print_help()
        return 0

    # Print logo before running command
    print(ATHENA_LOGO)

    # Run the appropriate command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
