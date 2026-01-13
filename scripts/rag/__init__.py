# -*- coding: utf-8 -*-
"""
RAG (Retrieval-Augmented Generation) Module for WCHW Dataset

Provides few-shot learning via similar problem retrieval.
"""

from .retriever import (
    WCHWRetriever,
    get_retriever,
    reset_retriever
)

__all__ = [
    'WCHWRetriever',
    'get_retriever',
    'reset_retriever',
]
