# Author.py
from __future__ import annotations

class Author:
    """Représente un auteur du corpus."""

    def __init__(self, name: str):
        self.name: str = name
        self.ndoc: int = 0
        self.production: dict[int, object] = {}

    def add(self, doc_id: int, document: object) -> None:
        self.production[doc_id] = document
        self.ndoc = len(self.production)

    def get_taille_moyenne_document(self) -> float:
        if self.ndoc == 0:
            return 0.0
        total_caracteres = sum(len(getattr(doc, "texte", "") or "") for doc in self.production.values())
        return total_caracteres / self.ndoc

    def __str__(self) -> str:
        return f"Author: {self.name} (Documents: {self.ndoc})"

    def __repr__(self) -> str:
        return f"Author(name={self.name!r}, ndoc={self.ndoc})"
