"""
Diversity checker using semantic similarity.

Ensures generated samples are diverse and not too similar to each other.
Uses sentence embeddings to compute semantic similarity.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DiversityChecker:
    """
    Check diversity of generated samples using semantic similarity.

    Uses sentence embeddings to ensure samples are sufficiently different.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        use_local: bool = True
    ):
        """
        Initialize diversity checker.

        Args:
            model_name: Sentence transformer model name
            similarity_threshold: Maximum allowed similarity (0-1)
            use_local: Use local sentence-transformers vs API
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.use_local = use_local

        # Initialize model
        self.model = None
        if use_local:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence transformer: {model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                self.use_local = False

        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        if self.use_local and self.model:
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            # Fallback: simple character-based vector (not very good)
            embedding = self._simple_embedding(text)

        # Cache result
        self.embedding_cache[text] = embedding

        return embedding

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding (character frequency)."""
        # Create a simple 26-dimensional vector based on character frequency
        vector = np.zeros(26)
        text_lower = text.lower()

        for char in text_lower:
            if 'a' <= char <= 'z':
                vector[ord(char) - ord('a')] += 1

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )

        return float(similarity)

    def is_diverse(
        self,
        new_sample: str,
        existing_samples: List[str],
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Check if new sample is diverse enough compared to existing samples.

        Args:
            new_sample: New sample text
            existing_samples: List of existing sample texts
            threshold: Optional override for similarity threshold

        Returns:
            Tuple of (is_diverse, max_similarity)
        """
        if not existing_samples:
            return True, 0.0

        threshold = threshold or self.similarity_threshold

        # Compute similarity to each existing sample
        similarities = []
        for existing in existing_samples:
            sim = self.compute_similarity(new_sample, existing)
            similarities.append(sim)

        max_similarity = max(similarities)
        is_diverse = max_similarity < threshold

        if not is_diverse:
            logger.debug(
                f"Sample too similar: max_sim={max_similarity:.3f} "
                f"(threshold={threshold})"
            )

        return is_diverse, max_similarity

    def filter_duplicates(
        self,
        samples: List[Dict[str, Any]],
        text_key: str = "prompt",
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter out near-duplicate samples.

        Args:
            samples: List of samples
            text_key: Key to extract text for comparison
            threshold: Similarity threshold for duplicates

        Returns:
            Filtered list without near-duplicates
        """
        if not samples:
            return []

        threshold = threshold or self.similarity_threshold

        # Keep first sample
        filtered = [samples[0]]
        filtered_texts = [self._extract_text(samples[0], text_key)]

        for sample in samples[1:]:
            sample_text = self._extract_text(sample, text_key)

            is_diverse, max_sim = self.is_diverse(
                sample_text,
                filtered_texts,
                threshold
            )

            if is_diverse:
                filtered.append(sample)
                filtered_texts.append(sample_text)
            else:
                logger.debug(f"Filtered duplicate: similarity={max_sim:.3f}")

        removed_count = len(samples) - len(filtered)
        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} near-duplicates "
                f"({removed_count/len(samples)*100:.1f}%)"
            )

        return filtered

    def _extract_text(self, sample: Dict[str, Any], text_key: str) -> str:
        """Extract text from sample for comparison."""
        # Try direct key access
        if text_key in sample:
            text = sample[text_key]
            if isinstance(text, str):
                return text

        # Try nested access for structured data
        if "data" in sample:
            data = sample["data"]

            # For SFT messages format
            if "messages" in data:
                messages = data["messages"]
                if isinstance(messages, list):
                    # Concatenate user and assistant messages
                    texts = [
                        msg.get("content", "")
                        for msg in messages
                        if msg.get("role") in ("user", "assistant")
                    ]
                    return " ".join(texts)

            # For completion format
            if "prompt" in data:
                return str(data["prompt"])

            # For DPO format
            if text_key in data:
                return str(data[text_key])

        # Fallback: convert entire sample to string
        return str(sample)

    def get_diversity_score(
        self,
        samples: List[Dict[str, Any]],
        text_key: str = "prompt"
    ) -> float:
        """
        Calculate overall diversity score for a dataset.

        Args:
            samples: List of samples
            text_key: Key to extract text

        Returns:
            Diversity score (0-1, where 1 is most diverse)
        """
        if len(samples) < 2:
            return 1.0

        # Extract texts
        texts = [self._extract_text(s, text_key) for s in samples]

        # Compute pairwise similarities
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = self.compute_similarity(texts[i], texts[j])
                similarities.append(sim)

        # Diversity = 1 - average similarity
        avg_similarity = np.mean(similarities)
        diversity_score = 1.0 - avg_similarity

        logger.info(
            f"Dataset diversity: {diversity_score:.3f} "
            f"(avg_similarity: {avg_similarity:.3f})"
        )

        return float(diversity_score)

    def cluster_samples(
        self,
        samples: List[Dict[str, Any]],
        text_key: str = "prompt",
        n_clusters: int = 5
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster samples by semantic similarity.

        Args:
            samples: List of samples
            text_key: Key to extract text
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping cluster_id to samples
        """
        if len(samples) < n_clusters:
            # All in one cluster
            return {0: samples}

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.warning("sklearn not installed, cannot cluster")
            return {0: samples}

        # Get embeddings
        texts = [self._extract_text(s, text_key) for s in samples]
        embeddings = np.array([self.get_embedding(t) for t in texts])

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # Group by cluster
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for idx, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(samples[idx])

        logger.info(
            f"Clustered {len(samples)} samples into {len(clusters)} clusters"
        )

        return clusters
