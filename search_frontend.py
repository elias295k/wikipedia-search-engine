from flask import Flask, request, jsonify

from flask import Flask, request, jsonify
import re, math, gzip, pickle
from collections import Counter, defaultdict
import csv, io

import numpy as np
from google.cloud import storage
from nltk.stem.porter import PorterStemmer

from inverted_index_gcp import InvertedIndex, MultiFileReader, TUPLE_SIZE, PROJECT_ID


# =============================================================================
# Config
# =============================================================================
BUCKET_NAME = "bgu-212741532"

# BODY (NO STEM)
BODY_NOSTEM_DIR = "postings_gcp"
BODY_NOSTEM_INDEX_NAME = "index"

# TITLE
TITLE_DIR = "postings_title_stem"
TITLE_INDEX_NAME = "index"

# ANCHOR (NO STEM)
ANCHOR_DIR = "postings_anchor"
ANCHOR_INDEX_NAME = "index"

# Meta
DOCID2IDX_GCS_PATH = "meta/docid2idx.pkl.gz"
ID2TITLE_GCS_PATH = "meta/id2title.pkl.gz"

# PageRank
PAGERANK_PREFIX = "pr/"

# Weights
ALPHA = 0.7  # body
BETA = 0.4    # title
DELTA = 0.4   # anchor
GAMMA = 0.1   # pagerank


# =============================================================================
# Tokenizer + Stopwords (hard-coded)
# =============================================================================
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

english_stopwords = frozenset([
    "during", "as", "whom", "no", "so", "shouldn't", "she's", "were", "needn", "then", "on",
    "should've", "once", "very", "any", "they've", "it's", "it", "be", "why", "ma", "over",
    "you'll", "they", "you've", "am", "before", "shan", "nor", "she'd", "because", "been",
    "doesn't", "than", "will", "they'd", "not", "those", "had", "this", "through", "again",
    "ours", "having", "himself", "into", "i'm", "did", "hadn", "haven", "should", "above",
    "we've", "does", "now", "m", "down", "he'd", "herself", "t", "their", "hasn't", "few",
    "and", "mightn't", "some", "do", "the", "we're", "myself", "i'd", "won", "after",
    "needn't", "wasn't", "them", "don", "further", "we'll", "hasn", "haven't", "out", "where",
    "mustn't", "won't", "at", "against", "shan't", "has", "all", "s", "being", "he'll", "he",
    "its", "that", "more", "by", "who", "i've", "o", "that'll", "there", "too", "they'll",
    "own", "aren't", "other", "an", "here", "between", "hadn't", "isn't", "below", "yourselves",
    "ve", "isn", "wouldn", "d", "we", "couldn", "ain", "his", "wouldn't", "was", "didn", "what",
    "when", "i", "i'll", "with", "her", "same", "you're", "yours", "couldn't", "for", "doing",
    "each", "aren", "which", "such", "mightn", "up", "mustn", "you", "only", "most", "of", "me",
    "she", "he's", "in", "a", "if", "but", "these", "him", "hers", "both", "my", "she'll", "re",
    "weren", "yourself", "is", "until", "weren't", "to", "are", "itself", "you'd", "themselves",
    "ourselves", "just", "wasn", "have", "don't", "ll", "how", "they're", "about", "shouldn",
    "can", "our", "we'd", "from", "it'd", "under", "while", "off", "y", "doesn", "theirs",
    "didn't", "or", "your", "it'll"
])

corpus_stopwords = frozenset([
    "category", "references", "also", "external", "links",
    "may", "first", "see", "history", "people", "one", "two",
    "part", "thumb", "including", "second", "following",
    "many", "however", "would", "became"
])

ALL_STOPWORDS = english_stopwords.union(corpus_stopwords)
STEMMER = PorterStemmer()


def tokenize_query_nostem(q: str):
    tokens = [m.group() for m in RE_WORD.finditer((q or "").lower())]
    out = []
    for t in tokens:
        if t in ALL_STOPWORDS:
            continue
        if len(t) < 2:
            continue
        out.append(t)
    return out


def tokenize_query_stem(q: str):
    tokens = [m.group() for m in RE_WORD.finditer((q or "").lower())]
    out = []
    for t in tokens:
        if t in ALL_STOPWORDS:
            continue
        t = STEMMER.stem(t)
        if len(t) < 2:
            continue
        if t in ALL_STOPWORDS:
            continue
        out.append(t)
    return out


# =============================================================================
# GCS helpers
# =============================================================================
_gcs_client = storage.Client(PROJECT_ID)
_bucket = _gcs_client.bucket(BUCKET_NAME)


def load_pickle_gz_from_gcs(path: str):
    blob = _bucket.blob(path)
    with blob.open("rb") as f:
        with gzip.GzipFile(fileobj=f, mode="rb") as gz:
            return pickle.load(gz)


# =============================================================================
# Load indexes + meta ONCE (module globals)
# =============================================================================
IDX_BODY_NOSTEM = InvertedIndex.read_index(
    BODY_NOSTEM_DIR, BODY_NOSTEM_INDEX_NAME, bucket_name=BUCKET_NAME
)
IDX_TITLE = InvertedIndex.read_index(
    TITLE_DIR, TITLE_INDEX_NAME, bucket_name=BUCKET_NAME
)
IDX_ANCHOR = InvertedIndex.read_index(
    ANCHOR_DIR, ANCHOR_INDEX_NAME, bucket_name=BUCKET_NAME
)

DOCID2IDX = load_pickle_gz_from_gcs(DOCID2IDX_GCS_PATH)
N_DOCS = len(DOCID2IDX)

try:
    ID2TITLE = load_pickle_gz_from_gcs(ID2TITLE_GCS_PATH)
except Exception:
    ID2TITLE = {}


# =============================================================================
# PageRank (optional, load once)
# =============================================================================
def load_pagerank_from_gcs(bucket, prefix: str):
    """
    Reads all *.csv.gz under gs://BUCKET/prefix and returns dict: doc_id -> pagerank(float)
    Assumes header row exists.
    """
    pr = {}
    for blob in bucket.list_blobs(prefix=prefix):
        name = blob.name
        if not name.endswith(".csv.gz"):
            continue
        data = blob.download_as_bytes()
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as gz:
            text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
            reader = csv.reader(text)
            _ = next(reader, None)  # header
            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    doc_id = int(row[0])
                    score = float(row[1])
                    pr[doc_id] = score
                except Exception:
                    continue
    return pr


try:
    PAGERANK = load_pagerank_from_gcs(_bucket, PAGERANK_PREFIX)
except Exception:
    PAGERANK = {}

PR_MAX = max(PAGERANK.values()) if PAGERANK else 1.0
LOG_PR_MAX = math.log1p(PR_MAX) if PR_MAX > 0 else 1.0


def pr_feature(doc_id: int) -> float:
    """
    Normalized [0,1]-ish: log1p(pr)/log1p(max_pr)
    """
    pr = PAGERANK.get(doc_id, 0.0)
    if pr <= 0.0:
        return 0.0
    return math.log1p(pr) / LOG_PR_MAX


# =============================================================================
# Posting list reader
# =============================================================================
def read_posting_list_with_reader(idx: InvertedIndex, term: str, reader: MultiFileReader):
    """
    Returns posting list for term: [(doc_id, tf), ...]
    """
    if term not in idx.posting_locs:
        return []
    df = idx.df.get(term, 0)
    if df <= 0:
        return []
    locs = idx.posting_locs[term]
    b = reader.read(locs, df * TUPLE_SIZE)

    pl = []
    for i in range(df):
        base = i * TUPLE_SIZE
        doc_id = int.from_bytes(b[base: base + 4], "big")
        tf = int.from_bytes(b[base + 4: base + TUPLE_SIZE], "big")
        pl.append((doc_id, tf))
    return pl


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # BODY query tokens (NO STEM)
    q_tokens_body = tokenize_query_nostem(query)
    if not q_tokens_body:
        return jsonify(res)

    q_tf = Counter(q_tokens_body)

    # Query weights: TF-IDF (BODY only; no norms)
    q_weights = {}
    for t, tfq in q_tf.items():
        df = IDX_BODY_NOSTEM.df.get(t, 0)
        if df <= 0:
            continue
        idf = math.log(N_DOCS / df)
        q_weights[t] = math.log1p(tfq) * idf

    if not q_weights:
        return jsonify(res)

    # BODY scoring: dot product (no cosine, no norms)
    doc_score = defaultdict(float)
    reader = MultiFileReader(BODY_NOSTEM_DIR, bucket_name=BUCKET_NAME)
    try:
        for t, wq in q_weights.items():
            pl = read_posting_list_with_reader(IDX_BODY_NOSTEM, t, reader)
            if not pl:
                continue

            df = IDX_BODY_NOSTEM.df.get(t, 0)
            if df <= 0:
                continue
            idf = math.log(N_DOCS / df)

            for doc_id, tfd in pl:
                wd = math.log1p(tfd) * idf
                doc_score[doc_id] += wq * wd
    finally:
        reader.close()

    if not doc_score:
        return jsonify(res)

    # TITLE boost: title index is stemmed -> stem query for title matching
    q_terms_title = set(tokenize_query_stem(query))
    title_match_count = defaultdict(int)
    if q_terms_title:
        # IMPORTANT FIX: posting_locs already includes directory prefix, so base_dir must be ""
        reader = MultiFileReader("", bucket_name=BUCKET_NAME)
        try:
            for t in q_terms_title:
                pl = read_posting_list_with_reader(IDX_TITLE, t, reader)
                for doc_id, _tf in pl:
                    title_match_count[doc_id] += 1
        finally:
            reader.close()
    denom_terms_title = max(1, len(q_terms_title))

    # ANCHOR boost: anchor index is NO-STEM -> use NO-STEM tokens
    q_terms_anchor = set(q_tokens_body)
    anchor_match_count = defaultdict(int)
    if q_terms_anchor:
        # IMPORTANT FIX: same reason as title
        reader = MultiFileReader("", bucket_name=BUCKET_NAME)
        try:
            for t in q_terms_anchor:
                pl = read_posting_list_with_reader(IDX_ANCHOR, t, reader)
                for doc_id, _tf in pl:
                    anchor_match_count[doc_id] += 1
        finally:
            reader.close()
    denom_terms_anchor = max(1, len(q_terms_anchor))

    # Combine (rank only docs that had BODY score)
    final = []
    for doc_id, bs in doc_score.items():
        ts = title_match_count.get(doc_id, 0) / denom_terms_title
        ans = anchor_match_count.get(doc_id, 0) / denom_terms_anchor
        prs = pr_feature(doc_id)

        final_score = (ALPHA * bs) + (BETA * ts) + (DELTA * ans) + (GAMMA * prs)
        final.append((doc_id, final_score))

    final.sort(key=lambda x: x[1], reverse=True)
    final = final[:100]

    for doc_id, _ in final:
        res.append((int(doc_id), ID2TITLE.get(int(doc_id), "")))


    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
