import os
from typing import List

TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", False)


def get_allowed_tokens() -> List[str] | None:
    if (
        tokens := os.getenv("AUTHENTICATION_ALLOWED_TOKENS", "").strip()
    ) and tokens != "":
        return tokens.strip().split(",")
