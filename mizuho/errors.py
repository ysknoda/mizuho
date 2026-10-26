"""The error a wrong input raises (ADR-0034)."""


class ConfigError(ValueError):
    """An input the config states is missing, malformed or inconsistent.

    `mizuho` prints it as one `ERROR:` line and exits 1; any other exception
    is a defect and keeps its traceback. A ValueError subclass, so callers
    that catch ValueError still catch it."""
