def __getattr__(name: str) -> object:
    if name == "GPSurrogate":
        from saealib.surrogate._deprecated import GPSurrogate

        return GPSurrogate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
