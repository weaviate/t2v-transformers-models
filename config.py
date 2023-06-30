import os
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

        # Split in sentences as long as T2V_TRANSFORMERS_DIRECT_TOKENIZE is not set to true or 1
        shall_split_in_sentences = os.getenv(
            "T2V_TRANSFORMERS_DIRECT_TOKENIZE", "false"
        ) not in ["true", "1"]

        return cls(
            cuda_config=cuda_config,
            shall_split_in_sentences=shall_split_in_sentences,
        )
