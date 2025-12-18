# Corpus.py
from __future__ import annotations

import pickle
import json
from typing import Optional, Dict, Any, List

import pandas as pd

from Author import Author
from Document import Document, RedditDocument, ArxivDocument, _to_datetime


class Corpus:
    """Gestion d'un corpus (TD4) + mise à jour TD5."""

    # Singleton (TD5 part 4.1)
    _singleton_instance: Optional["Corpus"] = None

    @classmethod
    def get_instance(cls, nom: str = "Corpus") -> "Corpus":
        if cls._singleton_instance is None:
            cls._singleton_instance = cls(nom)
        return cls._singleton_instance

    def __init__(self, nom: str):
        self.nom: str = nom
        self.authors: Dict[str, Author] = {}
        self.id2doc: Dict[int, Document] = {}
        self.ndoc: int = 0
        self.naut: int = 0
        self._next_id: int = 0  # TD4 1.3

    def add_document(self, document: Document) -> int:
        doc_id = self._next_id
        self._next_id += 1

        self.id2doc[doc_id] = document
        self.ndoc = len(self.id2doc)

        auteur_nom = document.auteur
        if auteur_nom not in self.authors:
            self.authors[auteur_nom] = Author(auteur_nom)
            self.naut = len(self.authors)

        self.authors[auteur_nom].add(doc_id, document)
        return doc_id

    # TD4 3.2
    def afficher_par_date(self, n: Optional[int] = None) -> None:
        items = sorted(self.id2doc.items(), key=lambda kv: kv[1].date)
        if n is None:
            n = len(items)
        print(f"=== Corpus '{self.nom}' — tri par date ===")
        for doc_id, doc in items[:n]:
            print(f"{doc.date.date()} (id={doc_id}) -> {doc}")

    def afficher_par_titre(self, n: Optional[int] = None) -> None:
        items = sorted(self.id2doc.items(), key=lambda kv: kv[1].titre.lower())
        if n is None:
            n = len(items)
        print(f"=== Corpus '{self.nom}' — tri par titre ===")
        for doc_id, doc in items[:n]:
            print(f"{doc.titre} (id={doc_id}) -> {doc}")

    def __repr__(self) -> str:
        return f"Corpus(nom={self.nom!r}, ndoc={self.ndoc}, naut={self.naut})"

    # TD4 3.3
    def to_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for doc_id, doc in self.id2doc.items():
            row: Dict[str, Any] = {
                "id": doc_id,
                "type": doc.getType(),          # TD5 3.2
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
            df = self.to_dataframe()
            df.to_csv(filename, sep="\t", index=False)
        elif format_type == "pickle":
            with open(filename, "wb") as f:
                pickle.dump(self, f)
        elif format_type == "json":
            payload = {
                "nom": self.nom,
                "documents": [
                    {
                        "id": doc_id,
                        "type": doc.getType(),
                        "titre": doc.titre,
                        "auteur": doc.auteur,
                        "date": doc.date.isoformat(),
                        "url": doc.url,
                        "texte": doc.texte,
                        "nb_commentaires": getattr(doc, "nb_commentaires", None),
                        "co_auteurs": getattr(doc, "co_auteurs", None),
                    }
                    for doc_id, doc in self.id2doc.items()
                ],
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError("format_type doit être 'csv', 'pickle' ou 'json'")

    @classmethod
    def load(cls, nom: str, filename: str, format_type: str = "csv") -> "Corpus":
        format_type = format_type.lower()

        if format_type == "pickle":
            with open(filename, "rb") as f:
                return pickle.load(f)

        if format_type == "json":
            with open(filename, "r", encoding="utf-8") as f:
                payload = json.load(f)
            corpus = cls(payload.get("nom", nom))
            for d in payload.get("documents", []):
                doc_type = (d.get("type") or "Document")
                titre = d.get("titre", "")
                auteur = d.get("auteur", "")
                date_val = _to_datetime(d.get("date"))
                url = d.get("url", "")
                texte = d.get("texte", "")
                if doc_type == "Reddit":
                    doc = RedditDocument(titre, auteur, date_val, url, texte, nb_commentaires=int(d.get("nb_commentaires") or 0))
                elif doc_type == "Arxiv":
                    co = d.get("co_auteurs") or []
                    doc = ArxivDocument(titre, auteur, date_val, url, texte, co_auteurs=co)
                else:
                    doc = Document(titre, auteur, date_val, url, texte)
                corpus.add_document(doc)
            return corpus

        # CSV/TSV
        df = pd.read_csv(filename, sep="\t")
        corpus = cls(nom)

        for _, row in df.iterrows():
            doc_type = str(row.get("type", "Document"))
            titre = str(row.get("titre", ""))
            auteur = str(row.get("auteur", ""))
            date_val = _to_datetime(str(row.get("date", "")))
            url = str(row.get("url", ""))
            texte = str(row.get("texte", ""))

            if doc_type == "Reddit":
                nb_com = int(row.get("nb_commentaires", 0) if not pd.isna(row.get("nb_commentaires", 0)) else 0)
                doc = RedditDocument(titre, auteur, date_val, url, texte, nb_commentaires=nb_com)
            elif doc_type == "Arxiv":
                co_str = "" if pd.isna(row.get("co_auteurs", "")) else str(row.get("co_auteurs", ""))
                co_auteurs = [x for x in co_str.split(";") if x] if co_str else []
                doc = ArxivDocument(titre, auteur, date_val, url, texte, co_auteurs=co_auteurs)
            else:
                doc = Document(titre, auteur, date_val, url, texte)

            corpus.add_document(doc)

        return corpus
