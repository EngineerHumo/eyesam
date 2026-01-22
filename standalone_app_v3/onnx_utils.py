import logging
from typing import List

import onnxruntime as ort

LOGGER = logging.getLogger(__name__)


def resolve_onnx_providers(prefer_gpu: bool = True) -> List[str]:
    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    LOGGER.info("ONNX Runtime providers=%s available=%s", providers, available)
    return providers
