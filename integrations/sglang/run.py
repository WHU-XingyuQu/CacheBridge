"""Compatibility launcher; prefer cachebridge serve."""

from cachebridge.backends.sglang.launcher import main

if __name__ == "__main__":
    main()
