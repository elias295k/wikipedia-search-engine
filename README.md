# wikipedia-search-engine

The engine supports multi-field ranking using:

Body (no stemming)

Title (stemmed)

Anchor text (no stemming)

PageRank boosting

Ranking Strategy

Each document receives a combined score:

Score = α·Body + β·Title + δ·Anchor + γ·PageRank

Weight	Field
α = 0.7	Body relevance
β = 0.4	Title relevance
δ = 0.4	Anchor relevance
γ = 0.1	PageRank boost

Storage Bucket

All posting lists and metadata are stored in:

gs://bgu-212741532


Directories used:

postings_gcp/          # Body (no-stem)
postings_title_stem/   # Title (stemmed)
postings_anchor/       # Anchor text
meta/                  # Norms, titles, and doc mappings
pr/                    # PageRank files

Query Processing

The engine supports two types of tokenization:

Function	Purpose
tokenize_query_nostem()	Used for body and anchor matching
tokenize_query_stem()	Used for title matching

Both functions:

Lowercase the query

Remove stopwords

Keep valid word tokens

Apply stemming only when needed

How Body Score is Computed

Count term frequencies in the query.

Compute query TF-IDF weights.

Read the posting list for each term from the body index.

Compute a dot product between query and document TF-IDF weights.

This produces a relevance score for each document.

Title and Anchor Boost

For each document:

Title score counts how many distinct query tokens appear in the title.

Anchor score counts how many distinct query tokens appear in anchor text.

Both are normalized by the number of query terms.

PageRank Integration

PageRank is loaded from GCS and normalized using:

log(1 + PageRank(doc)) / log(1 + maxPageRank)


This prevents extremely large PageRank values from dominating the ranking.

Output Format

Each search returns:

[
  (wiki_id, article_title),
  ...
]


Sorted from best to worst match, with a maximum of 100 results.
