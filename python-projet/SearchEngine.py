# SearchEngine.py

import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from Corpus import Corpus


class SearchEngine:
    """
    TD7 Search Engine
    - takes a Corpus in constructor
    - builds vocab + mat_TF + mat_TFxIDF immediately
    - provides search(query, top_n) returning a pandas DataFrame
    """

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.doc_ids = sorted(corpus.id2doc.keys())
        self.N = len(self.doc_ids)

        # Build vocab and matrices
        self.vocab = {}  # word -> {"id": int, "tf": int, "df": int, "idf": float}
        self.mat_TF = None
        self.mat_TFxIDF = None

        self._build()

    def _tokenize(self, text: str):
        cleaned = self.corpus.nettoyer_texte(text)
        return cleaned.split() if cleaned else []

    def _build(self):
        # 1) Build vocabulary + per-doc counts 
        word_set = set()
        docs_tokens = []

        for doc_id in self.doc_ids:
            tokens = self._tokenize(self.corpus.id2doc[doc_id].texte)
            docs_tokens.append(tokens)
            word_set.update(tokens)

        # Sort alphabetically
        words = sorted(word_set)
        for idx, w in enumerate(words):
            self.vocab[w] = {"id": idx, "tf": 0, "df": 0, "idf": 0.0}

        # 2) Build sparse TF matrix
        rows = []
        cols = []
        data = []

        for i, tokens in enumerate(docs_tokens):
            if not tokens:
                continue

            local_counts = {}
            for w in tokens:
                local_counts[w] = local_counts.get(w, 0) + 1

            # doc frequency update (once per doc)
            for w in local_counts.keys():
                self.vocab[w]["df"] += 1

            # fill sparse data
            for w, c in local_counts.items():
                j = self.vocab[w]["id"]
                rows.append(i)
                cols.append(j)
                data.append(c)

        V = len(self.vocab)
        self.mat_TF = csr_matrix((data, (rows, cols)), shape=(self.N, V), dtype=float)

        # 3) Compute total term frequency per word (corpus tf)
        # sum over docs for each column
        col_sums = np.asarray(self.mat_TF.sum(axis=0)).ravel()
        for w, info in self.vocab.items():
            info["tf"] = int(col_sums[info["id"]])

        # 4) Compute IDF + TFxIDF matrix
        # idf = log((N + 1) / (df + 1)) + 1  (smooth)
        idf = np.zeros(V, dtype=float)
        for w, info in self.vocab.items():
            df = info["df"]
            val = math.log((self.N + 1) / (df + 1)) + 1.0
            info["idf"] = float(val)
            idf[info["id"]] = val

        # multiply each column by its idf
        self.mat_TFxIDF = self.mat_TF.multiply(idf)

    def _query_vector(self, query: str, use_tfidf: bool = True):
        tokens = self._tokenize(query)
        if not tokens:
            return None

        V = len(self.vocab)
        q = np.zeros(V, dtype=float)

        # TF in query
        for w in tokens:
            if w in self.vocab:
                j = self.vocab[w]["id"]
                q[j] += 1.0

        if use_tfidf:
            # multiply by idf
            for w in set(tokens):
                if w in self.vocab:
                    j = self.vocab[w]["id"]
                    q[j] *= self.vocab[w]["idf"]

        # normalize for cosine similarity
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        return q

    def search(self, keywords: str, top_n: int = 10, use_tfidf: bool = True) -> pd.DataFrame:
        """
        TD7: returns a pandas DataFrame.
        Similarity: cosine similarity with doc vectors.
        """
        if self.N == 0:
            return pd.DataFrame(columns=["doc_id", "score", "titre", "auteur", "date", "type", "url"])

        q = self._query_vector(keywords, use_tfidf=use_tfidf)
        if q is None:
            return pd.DataFrame(columns=["doc_id", "score", "titre", "auteur", "date", "type", "url"])

        mat = self.mat_TFxIDF if use_tfidf else self.mat_TF

        # cosine: dot( doc_normed , q_normed )
        # normalize docs row-wise
        doc_norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        doc_norms[doc_norms == 0] = 1.0
        mat_normed = mat.multiply(1.0 / doc_norms[:, None])

        scores = mat_normed.dot(q)
        scores = np.asarray(scores).ravel()

        # rank
        order = np.argsort(-scores)[:top_n]

        rows = []
        for i in order:
            doc_id = self.doc_ids[i]
            doc = self.corpus.id2doc[doc_id]
            rows.append({
                "doc_id": doc_id,
                "score": float(scores[i]),
                "titre": doc.titre,
                "auteur": doc.auteur,
                "date": doc.date.isoformat(),
                "type": doc.getType(),
                "url": doc.url,
            })

        return pd.DataFrame(rows)
