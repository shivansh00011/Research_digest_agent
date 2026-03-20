"""Research Digest Agent Package"""

__version__ = "1.0.0"

from .agent import ResearchDigestAgent
from .ingestion import ContentIngester, SourceContent
from .extraction import ClaimExtractor, Claim
from .llm_extraction import GeminiClaimExtractor, HybridClaimExtractor
from .deduplication import ClaimDeduplicator, ClaimGroup
from .generation import DigestGenerator

__all__ = [
    "ResearchDigestAgent",
    "ContentIngester",
    "SourceContent",
    "ClaimExtractor",
    "Claim",
    "GeminiClaimExtractor",
    "HybridClaimExtractor",
    "ClaimDeduplicator",
    "ClaimGroup",
    "DigestGenerator",
]
