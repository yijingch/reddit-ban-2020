


def parse_years(spec: str) -> list[int]:
    """'2005:2019' -> inclusive range; '2018,2019' -> list."""
    if ":" in spec:
        start, end = spec.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def fill_path(path: str, subs: dict) -> str:
    """Substitute only known {placeholders}; leave anything else alone."""
    for key, val in subs.items():
        if val is not None:
            path = path.replace("{" + key + "}", str(val))
    return path

