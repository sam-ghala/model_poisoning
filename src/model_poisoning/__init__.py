"""
model_poisoning: Reproduction study of "Sleeper Agents" (arXiv:2401.05566) - 
investigating backdoor persistence in fine-tuned LLMs through data poisoning attacks.
"""

from __future__ import annotations

from importlib.metadata import version

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

__all__ = ("__version__",)
__version__ = version(__name__)
