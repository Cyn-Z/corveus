"""PowerShell inspection that decodes text but never executes it."""
from __future__ import annotations

import base64
import hashlib
import re

from dfir_contracts import DecodedCommand, NormalizedEvent

_ENCODED = re.compile(r"(?:^|\s)-(?:encodedcommand|enc)\s+([^\s]+)", re.IGNORECASE)
_POWERSHELL = ("powershell.exe", "pwsh.exe")


def decode_encoded_command(value: str) -> DecodedCommand:
    if not isinstance(value, str) or not value.strip():
        return DecodedCommand(str(value), None, None, False)
    try:
        raw = base64.b64decode(value, validate=True)
        text = raw.decode("utf-16le")
        if not text:
            raise ValueError
    except (ValueError, UnicodeDecodeError):
        return DecodedCommand(value, None, None, False)
    return DecodedCommand(value, text, "utf-16le", True, hashlib.sha256(text.encode()).hexdigest())


class PowerShellRecord:
    def __init__(self, event, encoded):
        self.event = event
        self.encoded = encoded


def inspect_powershell(events):
    records = []
    for event in events:
        image = (event.image or "").lower().replace("/", "\\")
        if not any(image.endswith(name) for name in _POWERSHELL):
            continue
        match = _ENCODED.search(event.command_line or "")
        encoded = decode_encoded_command(match.group(1)) if match else DecodedCommand("", None, None, False)
        records.append(PowerShellRecord(event, encoded))
    return tuple(records)
