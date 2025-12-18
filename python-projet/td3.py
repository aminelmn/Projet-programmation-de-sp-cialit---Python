# td3.py

import os
from datetime import datetime

import pandas as pd
import praw
import urllib.request
import urllib.parse
import xmltodict

from Document import Document, RedditDocument, ArxivDocument
from Corpus import Corpus


# Partie 1 : chargement données

keyword = "climate"  # thématique (anglais)

docs = []  # TD3 : liste de textes uniquement
rows = []  # table enrichie pour TD4/TD5


# --- Reddit ---
reddit = praw.Reddit(
    client_id="ibcuYi10uqxHHdXzB5uYKQ",
    client_secret="jtE5tO0PQEonBNHswyXmPJab-Oyn2w",
    user_agent="aminelmn project")

subreddit = reddit.subreddit("all")
for submission in subreddit.search(keyword, limit=10):
    if not submission.selftext:
        continue

    # contenu textuel (TD3)
    texte = submission.selftext.replace("\n", " ").strip()
    docs.append(texte)

    # métadonnées (TD4/TD5)
    titre = (submission.title or "").strip()
    auteur = str(submission.author) if submission.author else "unknown"
    date_pub = datetime.fromtimestamp(getattr(submission, "created_utc", datetime.now().timestamp()))
    url = getattr(submission, "url", "")
    nb_commentaires = int(getattr(submission, "num_comments", 0) or 0)

    rows.append({
        "origine": "reddit",
        "titre": titre if titre else f"reddit-{len(rows)+1}",
        "auteur": auteur,
        "date": date_pub,
        "url": url,
        "texte": texte,
        "nb_commentaires": nb_commentaires,
        "co_auteurs": ""
    })


# --- Arxiv ---
base_url = "http://export.arxiv.org/api/query?"
query = f"search_query=all:{urllib.parse.quote(keyword)}&start=0&max_results=10"
response = urllib.request.urlopen(base_url + query)
data = response.read()
parsed_data = xmltodict.parse(data)

entries = parsed_data.get("feed", {}).get("entry", [])
if isinstance(entries, dict):
    entries = [entries]

for entry in entries:
    summary = (entry.get("summary") or "").replace("\n", " ").strip()
    if not summary:
        continue

    # contenu textuel (TD3)
    docs.append(summary)

    # métadonnées (TD4/TD5)
    titre = (entry.get("title") or "").replace("\n", " ").strip()

    author_field = entry.get("author", {})
    if isinstance(author_field, list) and author_field:
        auteur = author_field[0].get("name", "unknown")
        co_auteurs = [a.get("name", "") for a in author_field[1:] if a.get("name")]
    elif isinstance(author_field, dict):
        auteur = author_field.get("name", "unknown")
        co_auteurs = []
    else:
        auteur = "unknown"
        co_auteurs = []

    published = entry.get("published") or entry.get("updated") or ""
    date_pub = datetime.fromisoformat(str(published).replace("Z", "")) if published else datetime.now()
    url = entry.get("id") or ""

    rows.append({
        "origine": "arxiv",
        "titre": titre if titre else f"arxiv-{len(rows)+1}",
        "auteur": auteur,
        "date": date_pub,
        "url": url,
        "texte": summary,
        "nb_commentaires": 0,
        "co_auteurs": ";".join(co_auteurs)
    })


print(f"Taille de docs (textes) : {len(docs)}")


# Partie 2 : sauvegarde des données

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("Aucun document récupéré. Vérifiez vos clés API / mots-clés.")

df.insert(0, "id", range(1, len(df) + 1))

out_dir = "data"
os.makedirs(out_dir, exist_ok=True)

# TD3 2.2 : .csv avec tabulation \t (on utilise .tsv pour être clair)
tsv_path = os.path.join(out_dir, "td3_documents.tsv")
df[["id", "texte", "origine"]].to_csv(tsv_path, sep="\t", index=False)


tsv_full_path = os.path.join(out_dir, "td3_documents_full.tsv")
df.to_csv(tsv_full_path, sep="\t", index=False)

print(f"Enregistré {len(df)} documents :")
print(f" - {tsv_path} (id, texte, origine)")
print(f" - {tsv_full_path} (enrichi TD4/TD5)")


# Partie 2.3 : chargement sans APIs
df_loaded = pd.read_csv(tsv_path, sep="\t")
print("Chargement direct OK, head():")
print(df_loaded.head())


# Partie 3 : premières manipulations

# 3.1 Taille du corpus
print(f"3.1 — Taille du corpus : {len(df_loaded)} documents")

# 3.2 Nombre de mots et de phrases
def count_words(s: str) -> int:
    return len(str(s).split())

def count_sentences(s: str) -> int:
    return sum(1 for seg in str(s).split(".") if seg.strip())

df_loaded["n_words"] = df_loaded["texte"].apply(count_words)
df_loaded["n_sentences"] = df_loaded["texte"].apply(count_sentences)
print("3.2 — Aperçu compteurs :")
print(df_loaded[["id", "n_words", "n_sentences"]].head())

# 3.3 Supprimer documents < 20 caractères
before = len(df_loaded)
df_loaded = df_loaded[df_loaded["texte"].astype(str).str.len() >= 20].reset_index(drop=True)
after = len(df_loaded)
print(f"3.3 — supprimés : {before - after}, restants : {after}")

# 3.4 concaténer tous les documents
all_text = " ".join(df_loaded["texte"].astype(str).tolist())
all_text_path = os.path.join(out_dir, "td3_all_text.txt")
with open(all_text_path, "w", encoding="utf-8") as f:
    f.write(all_text)
print(f"3.4 — all_text enregistré : {all_text_path} (len={len(all_text)})")


# TD4/TD5 : instanciation Document/Corpus
df_full = pd.read_csv(tsv_full_path, sep="\t")

corpus = Corpus("Corpus TD3→TD5")
for _, r in df_full.iterrows():
    origine = str(r["origine"]).lower()
    titre = str(r.get("titre", ""))
    auteur = str(r.get("auteur", ""))
    date_pub = r.get("date", "")
    url = str(r.get("url", ""))
    texte = str(r.get("texte", ""))

    if origine == "reddit":
        nb_com = int(r.get("nb_commentaires", 0) if not pd.isna(r.get("nb_commentaires", 0)) else 0)
        doc = RedditDocument(titre, auteur, date_pub, url, texte, nb_commentaires=nb_com)
    elif origine == "arxiv":
        co_str = "" if pd.isna(r.get("co_auteurs", "")) else str(r.get("co_auteurs", ""))
        co_auteurs = [x for x in co_str.split(";") if x] if co_str else []
        doc = ArxivDocument(titre, auteur, date_pub, url, texte, co_auteurs=co_auteurs)
    else:
        doc = Document(titre, auteur, date_pub, url, texte)

    corpus.add_document(doc)

# TD4 3.2 + TD5 3.2
corpus.afficher_par_date(n=5)
corpus.afficher_par_titre(n=5)

# TD4 3.3-3.4
corpus_path = os.path.join(out_dir, "corpus_td4_td5.tsv")
corpus.save(corpus_path, format_type="csv")
corpus2 = Corpus.load("Corpus reload", corpus_path, format_type="csv")
print("Reload OK ->", corpus2)
