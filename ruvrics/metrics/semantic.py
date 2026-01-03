"""
Semantic consistency metric calculation.

Measures how similar outputs are in meaning using embeddings.
From spec Section 3.1 - Semantic Consistency.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from ruvrics.config import Config, get_config
from ruvrics.core.models import MetricResult
from ruvrics.utils.errors import EmbeddingError


class SemanticAnalyzer:
    """
    Analyzes semantic consistency using sentence embeddings.

    Uses centroid-based similarity (not pairwise) for efficiency.
    From spec Section 3.1.
    """

    def __init__(self, config: Config | None = None):
        """
        Initialize semantic analyzer.

        Args:
            config: Configuration (uses global if None)
        """
        self.config = config or get_config()

        try:
            # Load embedding model (from spec Appendix C)
            self.model = SentenceTransformer(self.config.embedding_model)
        except Exception as e:
            raise EmbeddingError(
                str(e), model=self.config.embedding_model.split("/")[-1]
            )

    def calculate_consistency(self, outputs: list[str]) -> MetricResult:
        """
        Calculate semantic consistency using centroid method.

        From spec Section 3.1:
        1. Embed all outputs
        2. Compute centroid embedding (mean)
        3. Calculate mean cosine similarity to centroid

        Args:
            outputs: List of output texts from all runs

        Returns:
            MetricResult with score (0-100) and variance classification

        Raises:
            EmbeddingError: If embedding fails
        """
        if len(outputs) < 2:
            raise ValueError("Need at least 2 outputs to calculate consistency")

        try:
            # Embed all outputs
            embeddings = self.model.encode(outputs)

            # Compute centroid (mean of all embeddings)
            centroid = np.mean(embeddings, axis=0)

            # Calculate cosine similarities to centroid
            similarities = []
            for emb in embeddings:
                # Cosine similarity: dot(a,b) / (norm(a) * norm(b))
                similarity = np.dot(emb, centroid) / (
                    np.linalg.norm(emb) * np.linalg.norm(centroid)
                )
                similarities.append(float(similarity))

            # Mean similarity (raw score in range ~0.7-1.0 for stable outputs)
            semantic_raw_score = np.mean(similarities)

            # Convert to 0-100 scale (from spec Section 3.1)
            semantic_consistency_score = semantic_raw_score * 100

            # Classify drift using thresholds from spec Section 4
            variance = self._classify_drift(semantic_consistency_score)

            return MetricResult(
                score=semantic_consistency_score,
                variance=variance,
                details={
                    "raw_score": semantic_raw_score,
                    "similarities": similarities,
                    "min_similarity": float(np.min(similarities)),
                    "max_similarity": float(np.max(similarities)),
                    "std_similarity": float(np.std(similarities)),
                },
            )

        except Exception as e:
            raise EmbeddingError(f"Failed to calculate semantic consistency: {e}")

    def _classify_drift(self, score: float) -> str:
        """
        Classify semantic drift based on score.

        From spec Section 4:
        - LOW: score >= 85
        - MEDIUM: 70 <= score < 85
        - HIGH: score < 70

        Args:
            score: Semantic consistency score (0-100)

        Returns:
            Drift classification: "LOW", "MEDIUM", or "HIGH"
        """
        if score >= self.config.semantic_low_threshold:  # 85
            return "LOW"
        elif score >= self.config.semantic_medium_threshold:  # 70
            return "MEDIUM"
        else:
            return "HIGH"


def get_embeddings(outputs: list[str]) -> np.ndarray:
    """
    Get embeddings for a list of outputs.

    Args:
        outputs: List of output texts

    Returns:
        Array of embeddings

    Raises:
        EmbeddingError: If embedding fails
    """
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(outputs)
        return embeddings
    except Exception as e:
        raise EmbeddingError(f"Failed to generate embeddings: {e}")


def calculate_semantic_consistency(
    outputs: list[str], config: Config | None = None
) -> MetricResult:
    """
    Convenience function to calculate semantic consistency.

    Args:
        outputs: List of output texts
        config: Optional configuration

    Returns:
        MetricResult with score and variance
    """
    analyzer = SemanticAnalyzer(config=config)
    return analyzer.calculate_consistency(outputs)
