import os
import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class CUDAConfig:
    enable_cuda: bool
    cuda_per_process_memory_fraction: Optional[float] = None
    cuda_core: Optional[str] = None


class T2VConfig:
    def __init__(
        self,
        cuda_config: CUDAConfig,
        shall_split_in_sentences: bool,
    ):
        self.cuda_config = cuda_config
        self.shall_split_in_sentences = shall_split_in_sentences

    @classmethod
    def from_env(cls):
        enable_cuda = os.getenv("ENABLE_CUDA") in ["true", "1"]
        cuda_config = CUDAConfig(enable_cuda=enable_cuda)

        if enable_cuda:
            cuda_config.cuda_core = os.getenv("CUDA_CORE", "cuda:0")

            try:
                cuda_config.cuda_per_process_memory_fraction = float(
                    os.getenv("CUDA_PER_PROCESS_MEMORY_FRACTION", 1.0)
                )
            except ValueError:
                raise ValueError(
                    "Invalid CUDA_PER_PROCESS_MEMORY_FRACTION (should be float)"
                )
            if not 0.0 <= cuda_config.cuda_per_process_memory_fraction <= 1.0:
                raise ValueError(
                    "Invalid CUDA_PER_PROCESS_MEMORY_FRACTION (should be between 0.0-1.0)"
                )

        shall_split_in_sentences = os.getenv("T2V_SHALL_SPLIT_IN_SENTENCES") in [
            "true",
            "1",
        ]

        direct_tokenize = os.getenv("T2V_DIRECT_TOKENIZE") in ["true", "1"]
        if direct_tokenize:
            shall_split_in_sentences = not direct_tokenize
            warnings.warn(
                "T2V_DIRECT_TOKENIZE will be deprecated in favour of T2V_SHALL_SPLIT_IN_SENTENCES",
                DeprecationWarning,
            )

        return cls(
            cuda_config=cuda_config,
            shall_split_in_sentences=shall_split_in_sentences,
        )
