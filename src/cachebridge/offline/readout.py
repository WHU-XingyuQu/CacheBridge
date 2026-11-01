"""Compatibility entry point for sequential V calibration."""


def main():
    from .calibration.sequential import main as run

    return run()


if __name__ == "__main__":
    main()
