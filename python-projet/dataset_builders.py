# dataset_builders.py
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from Corpus import Corpus
from Document import Document
from text_utils import split_sentences


def build_corpus_from_discours_us(
    path: str,
    corpus_name: str = "Discours US",
    limit_rows: int | None = None,
) -> Corpus:
    """
    TD8: load discours_US.csv (tab-separated), split each speech into sentences,
    each sentence becomes a Document.
    """
    df = pd.read_csv(path, sep="\t", engine="python")
    if limit_rows is not None:
        df = df.head(limit_rows)

    corpus = Corpus(corpus_name)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building corpus (sentences)"):
        speaker = str(row.get("speaker", "unknown") or "unknown")
        speech = str(row.get("text", "") or "")
        descr = str(row.get("descr", "") or "").strip()
        link = str(row.get("link", "") or "").strip()

        # date
        raw_date = str(row.get("date", "") or "").strip()
        dt = datetime.now()
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                dt = datetime.strptime(raw_date, fmt)
                break
            except ValueError:
                pass

        sentences = split_sentences(speech)
        for i, sent in enumerate(sentences):
            titre = descr if descr else f"Speech sentence #{i+1}"
            doc = Document(titre=titre, auteur=speaker, date=dt, url=link, texte=sent)
            corpus.add_document(doc)

    return corpus
