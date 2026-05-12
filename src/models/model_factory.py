"""Cached factory functions for encoder models.

Both ``get_bi_encoder`` and ``get_cross_encoder`` use ``functools.lru_cache``
so that calling them multiple times with the same arguments returns the same
object — avoiding repeated heavy model loads within a process.

Usage
-----
>>> from src.models.model_factory import get_bi_encoder, get_cross_encoder
>>> enc = get_bi_encoder()          # loads from src.config.settings
>>> enc2 = get_bi_encoder()         # instant — same object returned
>>> enc is enc2
True
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

from src.config import settings
from src.models.bi_encoder import BiEncoder
from src.models.cross_encoder import CrossEncoder

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def get_bi_encoder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> BiEncoder:
    """Return a cached :class:`BiEncoder` instance.

    Parameters
    ----------
    model_name:
        Hugging Face model ID.  Defaults to ``settings.bi_encoder_model``.
    device:
        PyTorch device string.  ``None`` lets sentence-transformers decide.

    Returns
    -------
    BiEncoder
        Shared (cached) instance for the given ``(model_name, device)`` key.

    Examples
    --------
    >>> enc = get_bi_encoder()
    >>> enc is get_bi_encoder()   # same object — no reload
    True
    """
    name = model_name or settings.bi_encoder_model
    logger.info("Factory: returning BiEncoder(%s)", name)
    return BiEncoder(model_name_or_path=name, device=device)


@functools.lru_cache(maxsize=4)
def get_cross_encoder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> CrossEncoder:
    """Return a cached :class:`CrossEncoder` instance.

    Parameters
    ----------
    model_name:
        Hugging Face model ID.  Defaults to
        ``settings.cross_encoder_model``.
    device:
        PyTorch device string.

    Returns
    -------
    CrossEncoder
        Shared (cached) instance for the given ``(model_name, device)`` key.

    Examples
    --------
    >>> reranker = get_cross_encoder()
    >>> reranker is get_cross_encoder()
    True
    """
    name = model_name or settings.cross_encoder_model
    logger.info("Factory: returning CrossEncoder(%s)", name)
    return CrossEncoder(model_name_or_path=name, device=device)


def clear_model_cache() -> None:
    """Evict all cached encoder instances.

    Useful in tests or when switching models at runtime without restarting
    the process.

    Examples
    --------
    >>> clear_model_cache()   # does not raise
    """
    get_bi_encoder.cache_clear()
    get_cross_encoder.cache_clear()
    logger.info("Model factory cache cleared")
