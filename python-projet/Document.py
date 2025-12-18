# Document.py
from __future__ import annotations

from datetime import datetime, date as date_class
from typing import Optional, List, Any


def _to_datetime(value: Any) -> datetime:
    """Convertit différentes représentations en datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date_class):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        s = value.strip()
        try:
            return datetime.fromisoformat(s.replace("Z", ""))
        except ValueError:
            pass
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return datetime.now()
    return datetime.now()


class Document:
    """Classe mère représentant un document générique."""

    def __init__(self, titre: str, auteur: str, date, url: str, texte: str):
        self.titre = titre
        self.auteur = auteur
        self.date: datetime = _to_datetime(date)
        self.url = url
        self.texte = texte
        self.type = "Document"  # champ demandé TD5

    def getType(self) -> str:
        return self.type

    def afficher(self) -> None:
        print(f"Titre : {self.titre}")
        print(f"Auteur : {self.auteur}")
        print(f"Date : {self.date}")
        print(f"URL : {self.url}")
        print(f"Texte : {self.texte}")

    def __str__(self) -> str:
        return f"[{self.getType()}] {self.titre}"

    def __repr__(self) -> str:
        return f"Document(titre={self.titre!r}, auteur={self.auteur!r}, date={self.date.isoformat()!r})"


class RedditDocument(Document):
    """Document Reddit (TD5). Champ spécifique : nb_commentaires."""

    def __init__(self, titre: str, auteur: str, date, url: str, texte: str, nb_commentaires: int = 0):
        super().__init__(titre, auteur, date, url, texte)
        self.type = "Reddit"
        self.nb_commentaires = int(nb_commentaires)

    def get_nb_commentaires(self) -> int:
        return self.nb_commentaires

    def set_nb_commentaires(self, nb_commentaires: int) -> None:
        nb_commentaires = int(nb_commentaires)
        if nb_commentaires < 0:
            raise ValueError("nb_commentaires doit être >= 0")
        self.nb_commentaires = nb_commentaires

    def getType(self) -> str:
        return "Reddit"

    def __str__(self) -> str:
        return f"[{self.getType()}] {self.titre} — commentaires : {self.nb_commentaires}"


class ArxivDocument(Document):
    """Document Arxiv (TD5). Champ spécifique : co_auteurs."""

    def __init__(self, titre: str, auteur: str, date, url: str, texte: str,
                 co_auteurs: Optional[List[str]] = None):
        super().__init__(titre, auteur, date, url, texte)
        self.type = "Arxiv"
        self.co_auteurs: List[str] = list(co_auteurs) if co_auteurs else []

    def get_co_auteurs(self) -> List[str]:
        return self.co_auteurs

    def set_co_auteurs(self, co_auteurs: List[str]) -> None:
        if not isinstance(co_auteurs, list):
            raise ValueError("co_auteurs doit être une liste")
        self.co_auteurs = [str(x) for x in co_auteurs]

    def getType(self) -> str:
        return "Arxiv"

    def __str__(self) -> str:
        co = ", ".join(self.co_auteurs) if self.co_auteurs else "Aucun"
        return f"[{self.getType()}] {self.titre} — co-auteurs : {co}"


class DocumentFactory:
    """Factory (patron d'usine) demandé TD5 (partie 4.2)."""

    @staticmethod
    def create(*, source: str, titre: str, auteur: str, date, url: str, texte: str,
               nb_commentaires: int = 0, co_auteurs: Optional[List[str]] = None) -> Document:
        s = (source or "").lower()
        if s == "reddit":
            return RedditDocument(titre, auteur, date, url, texte, nb_commentaires=nb_commentaires)
        if s == "arxiv":
            return ArxivDocument(titre, auteur, date, url, texte, co_auteurs=co_auteurs or [])
        return Document(titre, auteur, date, url, texte)
