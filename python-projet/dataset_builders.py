# dataset_builders.py
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from Corpus import Corpus
from Document import Document
from text_utils import split_sentences


def build_corpus_from_us_speeches_csv(
    csv_path: str,
    corpus_name: str = "Discours US",
    text_col: str = "text",
    author_col: str = "author",
    date_col: str = "date",
    url_col: str | None = None,
    limit_rows: int | None = None,
) -> Corpus:
    """
    TD8: load speeches CSV, split speeches into sentences -> each sentence becomes a Document.
    """
    df = pd.read_csv(csv_path)
    if limit_rows is not None:
        df = df.head(limit_rows)

    corpus = Corpus(corpus_name)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building corpus (sentences)"):
        speech = str(row.get(text_col, "") or "")
        author = str(row.get(author_col, "unknown") or "unknown")
        url = str(row.get(url_col, "")) if url_col else ""

        # date -> datetime
        raw_date = row.get(date_col, "")
        dt = None
        if isinstance(raw_date, datetime):
            dt = raw_date
        else:
            s = str(raw_date).strip()
            # try ISO first, else fallback
            try:
                dt = datetime.fromisoformat(s.replace("Z", ""))
            except Exception:
                # common fallback
                try:
                    dt = datetime.strptime(s[:10], "%Y-%m-%d")
                except Exception:
                    dt = datetime.now()

        # split into sentences
        sentences = split_sentences(speech)
        for i, sent in enumerate(sentences):
            titre = f"Speech sentence #{i+1}"
            doc = Document(titre=titre, auteur=author, date=dt, url=url, texte=sent)
            corpus.add_document(doc)

    return corpus
