# explorer.py
import math
import pandas as pd
from collections import Counter
from datetime import datetime

class Explorer:
    """
    TD9-10: higher-level exploration utilities:
    - compare two subcorpora (by type/source)
    - temporal evolution of a word/group
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def _tokens(self, text: str):
        cleaned = self.corpus.nettoyer_texte(text)
        return cleaned.split() if cleaned else []

    def compare_by_type(self, type_a: str, type_b: str, top_n: int = 20) -> pd.DataFrame:
        """
        Compare vocab between two doc types (Reddit vs Arxiv).
        Returns a dataframe with TF and relative TF.
        """
        a_counts = Counter()
        b_counts = Counter()
        a_total = 0
        b_total = 0

        for doc in self.corpus.id2doc.values():
            toks = self._tokens(doc.texte)
            if not toks:
                continue
            if doc.getType() == type_a:
                a_counts.update(toks)
                a_total += len(toks)
            elif doc.getType() == type_b:
                b_counts.update(toks)
                b_total += len(toks)

        rows = []
        vocab = set(a_counts) | set(b_counts)
        for w in vocab:
            a_tf = a_counts.get(w, 0)
            b_tf = b_counts.get(w, 0)
            a_rel = a_tf / a_total if a_total else 0.0
            b_rel = b_tf / b_total if b_total else 0.0
            rows.append({
                "mot": w,
                f"tf_{type_a}": a_tf,
                f"tf_{type_b}": b_tf,
                f"rel_{type_a}": a_rel,
                f"rel_{type_b}": b_rel,
                "diff_rel": a_rel - b_rel
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("diff_rel", ascending=False).head(top_n).reset_index(drop=True)
        return df

    def temporal_trend(self, term: str, freq: str = "M") -> pd.DataFrame:
        """
        Track relative frequency of 'term' through time.
        freq: 'M' monthly, 'Y' yearly, etc.
        """
        term = term.lower().strip()
        rows = []

        for doc in self.corpus.id2doc.values():
            dt = doc.date if isinstance(doc.date, datetime) else datetime.now()
            toks = self._tokens(doc.texte)
            if not toks:
                continue
            total = len(toks)
            hits = sum(1 for t in toks if t == term)
            rows.append({"date": dt, "hits": hits, "total": total})

        if not rows:
            return pd.DataFrame(columns=["period", "hits", "total", "rel_freq"])

        df = pd.DataFrame(rows)
        df["period"] = df["date"].dt.to_period(freq).dt.to_timestamp()
        agg = df.groupby("period", as_index=False)[["hits", "total"]].sum()
        agg["rel_freq"] = agg["hits"] / agg["total"].replace(0, 1)
        return agg
