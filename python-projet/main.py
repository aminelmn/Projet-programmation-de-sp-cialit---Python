# main.py
# v2 (TD3 -> TD7)

import os
from datetime import datetime
import pandas as pd

from Corpus import Corpus
from Document import Document, RedditDocument, ArxivDocument
from SearchEngine import SearchEngine


# Helpers: load data for v2

def load_corpus_from_tsv(tsv_path: str, corpus_name: str = "Corpus v2") -> Corpus:
    
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV introuvable: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")

    corpus = Corpus(corpus_name)

    # Case A: it's already a full corpus TSV
    if {"titre", "auteur", "date", "url", "texte"}.issubset(df.columns):
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

    # Case B: it's a texte + origine/source only TSV
    text_col = "texte" if "texte" in df.columns else ("text" if "text" in df.columns else None)
    src_col = "origine" if "origine" in df.columns else ("source" if "source" in df.columns else None)

    if text_col is None or src_col is None:
        raise ValueError(
            "TSV non reconnu. Attendu soit un corpus complet (titre,auteur,date,url,texte), "
            "soit un TSV TD3-like (texte/text + origine/source)."
        )

    for i, row in df.iterrows():
        texte = str(row.get(text_col, ""))
        origine = str(row.get(src_col, "")).lower().strip()

        # Minimal metadata
        titre = f"{origine.upper()} doc {i+1}"
        auteur = "unknown"
        url = ""
        dt = datetime.now()

        if origine == "reddit":
            doc = RedditDocument(titre, auteur, dt, url, texte, nb_commentaires=0)
        elif origine == "arxiv":
            doc = ArxivDocument(titre, auteur, dt, url, texte, co_auteurs=[])
        else:
            doc = Document(titre, auteur, dt, url, texte)

        corpus.add_document(doc)

    return corpus


# TD6 demo

def run_td6(corpus: Corpus):
    print("\n================ TD6 ================")

    # 1. Search (regex-based snippet search)
    kw = "climate"
    print(f"\n[TD6] search('{kw}') -> snippets (first 3):")
    snippets = corpus.search(kw)
    for s in snippets[:3]:
        print("-", s)

    # 2. Concordancer
    print(f"\n[TD6] concorde(r'\\b{kw}\\b', context=25) -> first 5 rows:")
    df_conc = corpus.concorde(rf"\b{kw}\b", context=25)
    print(df_conc.head(5))

    # 3. Stats (top words)
    print("\n[TD6] stats(top=10):")
    df_stats = corpus.stats(n=10)
    return df_stats


# TD7 demo

def run_td7(corpus: Corpus):
    print("\n================ TD7 ================")
    print("[TD7] Building SearchEngine (vocab + TF + TF-IDF matrices)...")
    engine = SearchEngine(corpus)

    query = "climate change"
    top_n = 5
    print(f"\n[TD7] search('{query}', top_n={top_n}, TF-IDF):")
    df_res = engine.search(query, top_n=top_n, use_tfidf=True)
    print(df_res)

    return df_res


# Main

def main():
    print("=== Projet Python v2 (TD3 -> TD7) ===")

    # Your file name:
    tsv_path = "corpus_td4_td5.tsv"

    # Load corpus
    corpus = load_corpus_from_tsv(tsv_path, corpus_name="Corpus v2 (TD3->TD7)")
    print(f"\n[INFO] Corpus chargé: ndoc={corpus.ndoc}, naut={corpus.naut}")

    # TD4 display
    print("\n[TD4] Aperçu tri par date (n=5):")
    corpus.afficher_par_date(n=5)

    print("\n[TD4] Aperçu tri par titre (n=5):")
    corpus.afficher_par_titre(n=5)

    # TD6
    run_td6(corpus)

    # TD7
    run_td7(corpus)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
