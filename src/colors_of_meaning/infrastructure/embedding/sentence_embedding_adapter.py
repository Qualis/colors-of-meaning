from typing import Any, Dict, List, Optional
import re
import numpy as np
import numpy.typing as npt


class SentenceEmbeddingAdapter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Optional[Any] = None

    def __getstate__(self) -> Dict[str, Any]:
        return {**self.__dict__, "_model": None}

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def encode(self, text: str) -> npt.NDArray:
        self._ensure_model_loaded()
        embedding = self._model.encode(text, convert_to_numpy=True)  # type: ignore
        return np.array(embedding, dtype=np.float32)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> npt.NDArray:
        self._ensure_model_loaded()
        embeddings = self._model.encode(  # type: ignore
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_document_sentences(self, document: str, batch_size: int = 32) -> npt.NDArray:
        sentences = self._split_into_sentences(document)
        return self.encode_batch(sentences, batch_size=batch_size)

    @staticmethod
    def _split_into_sentences(document: str) -> List[str]:
        pattern = r"[^.!?]*[.!?]"
        sentences = re.findall(pattern, document)
        remainder = re.sub(pattern, "", document).strip()

        result = [s.strip() for s in sentences]
        if remainder:
            result.append(remainder)

        return [s for s in result if s]

    @property
    def embedding_dimension(self) -> int:
        self._ensure_model_loaded()
        return int(self._model.get_sentence_embedding_dimension())  # type: ignore
