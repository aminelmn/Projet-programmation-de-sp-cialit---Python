# Corpus.py

import re
import pickle
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from Author import Author
from Document import Document, RedditDocument, ArxivDocument


class Corpus:
    def __init__(self, nom: str):
        self.nom = nom
        self.authors: Dict[str, Author] = {}
        self.id2doc: Dict[int, Document] = {}
        self.ndoc = 0
        self.naut = 0
        self._next_id = 0

        # TD6 cache: concatenated corpus string
        self._all_text_cache: Optional[str] = None

    # TD4/TD5: add documents
    def add_document(self, document: Document) -> int:
        doc_id = self._next_id
        self._next_id += 1

        self.id2doc[doc_id] = document
        self.ndoc = len(self.id2doc)

        a = document.auteur
        if a not in self.authors:
            self.authors[a] = Author(a)
            self.naut = len(self.authors)
        self.authors[a].add(doc_id, document)

        # invalidate cache
        self._all_text_cache = None
        return doc_id

    # TD4: sorting display
    def afficher_par_date(self, n: Optional[int] = None) -> None:
        items = sorted(self.id2doc.items(), key=lambda kv: kv[1].date)
        if n is None:
            n = len(items)
        for doc_id, doc in items[:n]:
            print(f"{doc.date.date()} (id={doc_id}) -> {doc}")

    def afficher_par_titre(self, n: Optional[int] = None) -> None:
        items = sorted(self.id2doc.items(), key=lambda kv: kv[1].titre.lower())
        if n is None:
            n = len(items)
        for doc_id, doc in items[:n]:
            print(f"{doc.titre} (id={doc_id}) -> {doc}")

    # Save/Load (TD4)
    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for doc_id, doc in self.id2doc.items():
            row = {
                "id": doc_id,
                "type": doc.getType(),
                "titre": doc.titre,
                "auteur": doc.auteur,
                "date": doc.date.isoformat(),
                "url": doc.url,
                "texte": doc.texte,
            }
            if isinstance(doc, RedditDocument):
                row["nb_commentaires"] = doc.nb_commentaires
            if isinstance(doc, ArxivDocument):
                row["co_auteurs"] = ";".join(doc.co_auteurs)
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, filename: str, format_type: str = "csv") -> None:
        format_type = format_type.lower()
        if format_type == "csv":
            self.to_dataframe().to_csv(filename, sep="\t", index=False)
        elif format_type == "pickle":
            with open(filename, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError("format_type must be 'csv' or 'pickle'")

    @classmethod
    def load(cls, nom: str, filename: str, format_type: str = "csv") -> "Corpus":
        format_type = format_type.lower()
        if format_type == "pickle":
            with open(filename, "rb") as f:
                return pickle.load(f)

        if format_type != "csv":
            raise ValueError("format_type must be 'csv' or 'pickle'")

        df = pd.read_csv(filename, sep="\t")
        corpus = cls(nom)

        for _, row in df.iterrows():
            doc_type = str(row.get("type", "Document"))
            titre = str(row.get("titre", ""))
            auteur = str(row.get("auteur", ""))
            url = str(row.get("url", ""))
            texte = str(row.get("texte", ""))

            date_str = str(row.get("date", ""))
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", ""))
            except ValueError:
                dt = datetime.now()

            if doc_type == "Reddit":
                nb = int(row.get("nb_commentaires", 0))
                doc = RedditDocument(titre, auteur, dt, url, texte, nb_commentaires=nb)
            elif doc_type == "Arxiv":
                co_str = str(row.get("co_auteurs", "")).strip()
                co = co_str.split(";") if co_str else []
                doc = ArxivDocument(titre, auteur, dt, url, texte, co_auteurs=co)
            else:
                doc = Document(titre, auteur, dt, url, texte)

            corpus.add_document(doc)

        return corpus

    # TD6

    def _build_all_text_once(self) -> str:
        """Build the concatenated corpus text only once, and cache it."""
        if self._all_text_cache is None:
            # concat all docs with separators to avoid merging words
            self._all_text_cache = "\n".join(
                str(doc.texte) for doc in self.id2doc.values()
            )
        return self._all_text_cache

    @staticmethod
    def nettoyer_texte(texte: str) -> str:
        """
        TD6 2.1: minimal cleaning:
        - lowercase
        - replace \n
        - optionally remove punctuation/digits via regex
        """
        t = (texte or "").lower().replace("\n", " ")
        # remove digits
        t = re.sub(r"\d+", " ", t)
        # replace punctuation with spaces
        t = re.sub(r"[^a-z\s]+", " ", t)
        # normalize whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def search(self, keyword: str, ignore_case: bool = True) -> List[str]:
        """
        TD6 1.1: return passages containing the keyword, by regex.
        Works on the concatenated string built once.
        Returns list of matched passages (snippets).
        """
        if not keyword:
            return []

        text = self._build_all_text_once()
        flags = re.IGNORECASE if ignore_case else 0

        # capture a small passage around the match
        pattern = re.compile(rf".{{0,40}}\b{re.escape(keyword)}\b.{{0,40}}", flags)
        return pattern.findall(text)

    def concorde(self, expr: str, context: int = 30, ignore_case: bool = True) -> pd.DataFrame:
        """
        TD6 1.2: Build a concordancer for an expression.
        Returns a pandas DataFrame with:
        contexte gauche | motif trouvé | contexte droit
        """
        if not expr:
            return pd.DataFrame(columns=["contexte gauche", "motif trouvé", "contexte droit"])

        text = self._build_all_text_once()
        flags = re.IGNORECASE if ignore_case else 0
        pattern = re.compile(expr, flags)

        rows = []
        for m in pattern.finditer(text):
            start, end = m.span()
            left = text[max(0, start - context):start]
            mid = text[start:end]
            right = text[end:min(len(text), end + context)]
            rows.append({
                "contexte gauche": left,
                "motif trouvé": mid,
                "contexte droit": right
            })

        return pd.DataFrame(rows, columns=["contexte gauche", "motif trouvé", "contexte droit"])

    def stats(self, n: int = 20) -> pd.DataFrame:
        """
        TD6 2.x:
        - number of distinct words
        - top-n most frequent words
        Builds a freq table using pandas.
        Returns the freq DataFrame (sorted).
        """
        # Build vocabulary + counts in one pass over documents 
        counts: Dict[str, int] = {}
        doc_freq: Dict[str, int] = {}

        for doc in self.id2doc.values():
            cleaned = self.nettoyer_texte(doc.texte)
            if not cleaned:
                continue
            tokens = cleaned.split()

            # term frequency
            for w in tokens:
                counts[w] = counts.get(w, 0) + 1

            # document frequency
            unique = set(tokens)
            for w in unique:
                doc_freq[w] = doc_freq.get(w, 0) + 1

        vocab_size = len(counts)
        print(f"Nombre de mots différents dans le corpus : {vocab_size}")

        freq_df = pd.DataFrame({
            "mot": list(counts.keys()),
            "tf": list(counts.values()),
            "df": [doc_freq.get(w, 0) for w in counts.keys()],
        })

        freq_df = freq_df.sort_values("tf", ascending=False).reset_index(drop=True)

        print(f"\nTop {n} mots les plus fréquents :")
        print(freq_df.head(n))

        return freq_df
