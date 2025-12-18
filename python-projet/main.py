# main.py
from datetime import datetime

from Document import DocumentFactory
from Corpus import Corpus


def main():
    # TD5 4.1 : Singleton (test)
    corpus = Corpus.get_instance("Mon premier corpus (Singleton)")

    # TD5 4.2 : Factory
    doc1 = DocumentFactory.create(
        source="reddit",
        titre="Post Reddit sur Python",
        auteur="Bob",
        date=datetime(2023, 5, 10),
        url="https://reddit.com/r/python/...",
        texte="Python est un langage très utilisé.",
        nb_commentaires=42
    )

    doc2 = DocumentFactory.create(
        source="arxiv",
        titre="Article Arxiv sur le deep learning",
        auteur="Carol",
        date=datetime(2022, 12, 1),
        url="https://arxiv.org/abs/1234.5678",
        texte="Cet article parle de deep learning.",
        co_auteurs=["Dave", "Eve"]
    )

    doc3 = DocumentFactory.create(
        source="document",
        titre="Un document générique",
        auteur="Alice",
        date=datetime(2024, 1, 15),
        url="https://exemple.com/doc1",
        texte="Ceci est le texte du premier document."
    )

    corpus.add_document(doc1)
    corpus.add_document(doc2)
    corpus.add_document(doc3)

    # TD4 3.2
    corpus.afficher_par_date()
    print()
    corpus.afficher_par_titre()

    # TD4 2.4
    auteur = corpus.authors.get("Carol")
    if auteur:
        print()
        print(f"Auteur : {auteur.name}")
        print(f"Nombre de documents : {auteur.ndoc}")
        print(f"Taille moyenne des documents : {auteur.get_taille_moyenne_document():.1f} caractères")

    # TD4 3.3-3.4
    corpus.save("corpus_td4_td5.tsv", format_type="csv")
    corpus_reload = Corpus.load("Corpus rechargé", "corpus_td4_td5.tsv", format_type="csv")
    print()
    print("Corpus rechargé :", corpus_reload)


if __name__ == "__main__":
    main()
