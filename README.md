# Product Recommendation System

A content-based product recommendation system built using sentence embeddings and cosine similarity on Amazon product data. Given a natural language query, it returns the most semantically relevant products from the catalog.

## What it does

The user types a product description (e.g. "wireless earbuds for running") and the system computes a similarity score between the query embedding and all pre-computed product embeddings. The top 10 matches are displayed as product cards with image, rating, price, and a direct link to Amazon.

There are three pages in the app:

- **Product Recommendation** - main search interface
- **Performance Metrics** - precision, recall, F1 score and similarity distribution charts
- **Model Accuracy** - evaluates the model on a held-out test sample

## Tech stack

- Python 3.9+
- Streamlit (UI)
- SentenceTransformers (all-MiniLM-L6-v2)
- scikit-learn (cosine similarity)
- pandas, numpy
- matplotlib (charts)

## Project structure

```
product_recommendation/
├── app.py                   # entry point, page routing
├── recommendation.py        # similarity search and product card rendering
├── data_processing.py       # data loading, preprocessing, embedding generation
├── performance_metrics.py   # chart functions for the metrics page
├── accuracy_metrics.py      # test set evaluation
├── .streamlit/
│   └── config.toml          # dark theme config
└── requirements.txt
```

The dataset files (`amazon_products.csv`, `amazon_categories.csv`) and the embedding file (`product_embeddings.npy`) are not included in the repository because of their size. See the setup section below.

## Setup

```bash
git clone https://github.com/Shreyasdotexe/PRODUCT-RECOMMENDATION-USING-AI
cd product_recommendation

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Place the following files in the project root before running:

- `amazon_products.csv`
- `amazon_categories.csv`
- `product_embeddings.npy` (optional — will be generated on first run if missing)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Notes on first run

If `product_embeddings.npy` is not present, the app will generate it automatically by encoding all product titles using the sentence transformer model. This takes around 10 to 30 minutes depending on hardware and is only done once.

## How similarity search works

1. The user's query is encoded into a 384-dimensional vector using the sentence transformer model.
2. Cosine similarity is computed between the query vector and all pre-computed product embeddings.
3. The top 10 products by similarity score are returned and displayed.

## Model accuracy evaluation

The accuracy page samples 5,000 products, splits them 80/20, and for each test embedding checks if any of the top-10 nearest training embeddings have cosine similarity >= 0.80. The fraction of test items that pass this check is reported as the accuracy.

## Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
sentence-transformers>=2.6.0
torch>=2.0.0
matplotlib>=3.7.0
```
red in `metrics.json` so the app loads them instantly without freezing.

## Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
sentence-transformers>=2.6.0
torch>=2.0.0
matplotlib>=3.7.0
```
