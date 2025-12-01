from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
import os

from dotenv import load_dotenv
from groq import Groq

from backend.pdf_parser import extract_text_from_pdf
from backend.chunker import chunk_text


load_dotenv()


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_id: int


class RagPipeline:
    """
    Jednostavan RAG sistem baziran na:
    - bag-of-words
    - TF-IDF ručnim vektorisanjem
    - Cosine-similarity retrieval
    - Groq LLaMA model odgovorima
    """

    def __init__(self):
        self.reset()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY nije definisan u okruženju.")

        self.llm = Groq(api_key=api_key)

    # ---------------------------------------
    # RESET PIPELINE
    # ---------------------------------------
    def reset(self):
        """Obriši sve što je vezano za prethodno indeksirane dokumente."""
        self.chunks: List[DocumentChunk] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.doc_matrix: Optional[np.ndarray] = None

    # ---------------------------------------
    # TOKENIZACIJA
    # ---------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\wšđčćž]+", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]

    # ---------------------------------------
    # INDEX BUILDING
    # ---------------------------------------
    def build_index(self, uploaded_files):
        all_chunks: List[DocumentChunk] = []

        for file in uploaded_files:
            name = file.name
            pages = extract_text_from_pdf(file.read())

            chunk_id = 0
            for page_num, text in pages:
                parts = chunk_text(text)
                for p in parts:
                    all_chunks.append(
                        DocumentChunk(
                            text=p,
                            source=name,
                            page=page_num,
                            chunk_id=chunk_id
                        )
                    )
                    chunk_id += 1

        if not all_chunks:
            raise ValueError("PDF dokument nema čitljiv tekst.")

        self.chunks = all_chunks

        # --- gradimo TF-IDF ---
        vocab = {}
        tokens_per_doc = []

        for ch in self.chunks:
            toks = self._tokenize(ch.text)
            tokens_per_doc.append(toks)
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)

        if not vocab:
            raise ValueError("Vokabular prazan – nema dovoljno teksta.")

        self.vocab = vocab

        N = len(self.chunks)
        V = len(vocab)

        tf = np.zeros((N, V), dtype=np.float32)
        df = np.zeros(V, dtype=np.int32)

        for i, toks in enumerate(tokens_per_doc):
            counts: Dict[int, int] = {}
            for t in toks:
                idx = vocab[t]
                counts[idx] = counts.get(idx, 0) + 1

            for idx, c in counts.items():
                tf[i, idx] = c
                df[idx] += 1

        # Normalizacija TF
        row_sum = tf.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        tf = tf / row_sum

        # IDF
        df[df == 0] = 1
        idf = np.log((1 + N) / df) + 1
        self.idf = idf

        # TF-IDF matrica
        tfidf = tf * idf

        # L2 norm
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tfidf = tfidf / norms

        self.doc_matrix = tfidf

    @property
    def is_ready(self):
        return self.doc_matrix is not None and len(self.chunks) > 0

    # ---------------------------------------
    # QUERY EMBEDDING
    # ---------------------------------------
    def _embed_query(self, q: str) -> np.ndarray:
        toks = self._tokenize(q)
        if not toks or not self.vocab:
            return np.zeros(len(self.vocab), dtype=np.float32)

        vec = np.zeros(len(self.vocab), dtype=np.float32)

        for t in toks:
            if t in self.vocab:
                vec[self.vocab[t]] += 1

        if vec.sum() > 0:
            vec /= vec.sum()

        if self.idf is not None:
            vec = vec * self.idf

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    # ---------------------------------------
    # RETRIEVAL
    # ---------------------------------------
    def _retrieve(self, q: str, top_k: int):
        qvec = self._embed_query(q)
        sims = self.doc_matrix @ qvec
        idxs = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in idxs]

    # ---------------------------------------
    # LLM ANSWER
    # ---------------------------------------
    def _llm_answer(self, question: str, context: str) -> str:
        prompt = f"""
Ti si AI sistem za pretragu dokumenata.

Kontekst:
{context}

Pitanje:
{question}

Uputstva:
- Koristi isključivo informacije iz konteksta.
- Ako informacija ne postoji, reci: "Informacija nije pronađena u dokumentu."
- Odgovaraj kratko i jasno.
"""

        resp = self.llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()

    # ---------------------------------------
    # MAIN ANSWER METHOD
    # ---------------------------------------
    def answer(self, question: str, top_k: int = 3):
        docs = self._retrieve(question, top_k)

        context_parts = []
        sources_meta = []

        for ch in docs:
            context_parts.append(f"[{ch.source} – strana {ch.page}]\n{ch.text}")

            sources_meta.append({
                "title": f"{ch.source} – strana {ch.page}",
                "snippet": ch.text[:350] + ("..." if len(ch.text) > 350 else "")
            })

        context = "\n\n".join(context_parts)

        answer = self._llm_answer(question, context)

        return answer, sources_meta
