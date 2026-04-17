import argparse
import os

from shiny import run_app

from configure_logger import LogLevel


def run_shiny(args: argparse.Namespace) -> None:
    kwargs: dict[str, object] = {"port": args.port, "host": args.host}
    if args.debug:
        kwargs.update({"reload": True, "reload_dirs": ["tfbpshiny/shiny_app"]})
    run_app("tfbpshiny.app:app", **kwargs)  # type: ignore


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tfbpshiny",
        description=(
            "tfbpshiny is a CLI with multiple utilities "
            "(e.g., shiny). Use --help after any command."
        ),
        epilog="Use 'tfbpshiny <utility> --help' for more info on each utility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Shared logging args
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )
    parser.add_argument(
        "--log-handler",
        type=str,
        default="console",
        choices=["console", "file"],
        help="Set log handler type",
    )
    parser.add_argument(
        "--profile-handler",
        type=str,
        default="console",
        choices=["console", "file"],
        help="Handler for the profiler logger",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable the profiler logger entirely",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: shiny
    shiny_parser = subparsers.add_parser("shiny", help="Run the shiny app")
    shiny_parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with auto-reload"
    )
    shiny_parser.add_argument(
        "--port", type=int, default=8000, help="Port to serve the Shiny app on"
    )
    shiny_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the Shiny app"
    )
    shiny_parser.set_defaults(func=run_shiny)

    # Example additional command:
    # another_parser = subparsers.add_parser("another", help="Another command")
    # another_parser.add_argument("--param", required=True)
    # another_parser.set_defaults(func=run_another_command)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    try:
        log_level = LogLevel.from_string(args.log_level)
        print(log_level)
    except ValueError as e:
        print(f"Invalid log level: {e}")
        parser.print_help()
        return

    # the log level is expected to be an int, but only str can be set as an env var
    # convert to int when configuring the logger
    os.environ["TFBPSHINY_LOG_LEVEL"] = str(log_level.value)
    os.environ["TFBPSHINY_LOG_HANDLER"] = args.log_handler
    os.environ["TFBPSHINY_PROFILE_HANDLER"] = args.profile_handler
    os.environ["TFBPSHINY_PROFILE_ENABLED"] = "0" if args.no_profile else "1"
    args.func(args)


if __name__ == "__main__":
    main()
