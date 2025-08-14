# parser.py
from typing import Optional

def detect_legacy_type(text: str, file_name: Optional[str] = None) -> str:
    fn = (file_name or "").lower()
    t = text.upper()
    if fn.endswith(".jcl") or ("//" in text and " EXEC " in t):
        return "JCL"
    if fn.endswith((".cbl", ".cob")) or "IDENTIFICATION DIVISION" in t:
        return "COBOL"
    return "TEXT"

def parse_code(raw_text: str, file_name: Optional[str] = None) -> str:
    """
    Annotate the source to help the LLM (non-destructive).
    Extend later: split divisions, extract copybooks, etc.
    """
    kind = detect_legacy_type(raw_text, file_name=file_name)
    header = f"[LEGACY-TYPE: {kind}] [FILENAME: {file_name or 'unknown'}]\n"
    return header + raw_text
