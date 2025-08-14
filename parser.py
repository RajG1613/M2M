def parse_code(legacy_code: str):
    # Simple example: remove empty lines
    return "\n".join(line for line in legacy_code.splitlines() if line.strip())
