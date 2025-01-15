import os
from typing import List

TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", False)


def get_allowed_tokens() -> List[str] | None:
    if (
        tokens := os.getenv("AUTHENTICATION_ALLOWED_TOKENS", "").strip()
    ) and tokens != "":
        return tokens.strip().split(",")


def get_use_sentence_transformers_multi_process():
    enable_multi_process = os.getenv("USE_SENTENCE_TRANSFORMERS_MULTI_PROCESS")
    if (
        enable_multi_process is not None
        and enable_multi_process == "true"
        or enable_multi_process == "1"
    ):
        return True
    return False


def get_t2v_transformers_direct_tokenize():
    transformers_direct_tokenize = os.getenv("T2V_TRANSFORMERS_DIRECT_TOKENIZE")
    if (
        transformers_direct_tokenize is not None
        and transformers_direct_tokenize == "true"
        or transformers_direct_tokenize == "1"
    ):
        return True
    return False
