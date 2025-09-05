# /core/worldgen/semantic_search.py

from ..common.utils import log_message

try:
    from sentence_transformers.util import cos_sim
except ImportError:
    cos_sim = None


class SemanticSearch:
    """
    A utility class for finding the best semantic match from a list of choices
    using a sentence embedding model.
    """

    def __init__(self, embedding_model):
        if embedding_model is None or cos_sim is None:
            log_message('debug',
                        "[SEMANTIC SEARCH] ERROR: Embedding model or sentence-transformers not available. Semantic fallback disabled.")
            self.model = None
        else:
            self.model = embedding_model

    def find_best_match(self, query: str, choices: list[str]) -> str | None:
        """
        Finds the most semantically similar string from a list of choices.

        Args:
            query: The string to match against.
            choices: A list of strings representing the valid options.

        Returns:
            The best matching string from the choices list, or None if matching is not possible.
        """
        if not self.model or not query or not choices:
            return None

        try:
            query_embedding = self.model.encode(query)
            choice_embeddings = self.model.encode(choices)

            # Compute cosine similarity
            similarities = cos_sim(query_embedding, choice_embeddings)

            # Find the index of the highest similarity score
            best_match_index = similarities.argmax()

            best_match = choices[best_match_index]
            score = similarities[0][best_match_index]

            log_message('debug', f"[SEMANTIC SEARCH] Matched query '{query}' to '{best_match}' with score {score:.4f}.")

            return best_match

        except Exception as e:
            log_message('debug', f"[SEMANTIC SEARCH] Error during embedding or similarity calculation: {e}")
            return None