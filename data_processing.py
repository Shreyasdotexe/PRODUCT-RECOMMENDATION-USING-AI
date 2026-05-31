import pandas as pd
import numpy as np
import os
import json
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── Device setup ──────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512 if DEVICE == 'cuda' else 64

# ── Model config ──────────────────────────────────────────────────────────────
# Upgraded from paraphrase-MiniLM-L3-v2 (3-layer) to all-MiniLM-L6-v2 (6-layer).
# Same inference speed class, significantly better semantic understanding.
MODEL_NAME = 'all-MiniLM-L6-v2'

# ── Deployment Mode ───────────────────────────────────────────────────────────
# Set to True to load the 50k Lite dataset (fits in 1GB RAM for free hosting)
# Set to False to load the full 1.4 million product dataset
USE_LITE_DATASET = True

if USE_LITE_DATASET:
    PRODUCTS_FILE = 'amazon_products_lite.csv'
    EMBEDDINGS_FILE = 'product_embeddings_lite.npy'
    METADATA_FILE = 'embeddings_metadata_lite.json'
else:
    PRODUCTS_FILE = 'amazon_products.csv'
    EMBEDDINGS_FILE = 'product_embeddings.npy'
    METADATA_FILE = 'embeddings_metadata.json'

TEXT_FORMAT_VERSION = 'title_category_v1'   # bump this if text construction changes

model = SentenceTransformer(MODEL_NAME, device=DEVICE)


def get_device_info():
    """Returns a human-readable device string for display in the sidebar."""
    if DEVICE == 'cuda':
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        return f"{name} ({vram} GB VRAM)", True
    return "CPU", False


def _make_product_text(row):
    """
    Builds the text string that gets embedded for each product.
    Including the category gives the model context beyond just the title.
    """
    title = str(row.get('title', '') or '').strip()
    category = str(row.get('category_name', '') or '').strip()
    if category and category not in ('Uncategorized', 'nan', ''):
        return f"{title} [{category}]"
    return title


def _embeddings_valid(num_products):
    """
    Checks whether the saved embeddings match the current model and text format.
    Returns False if they need to be regenerated.
    """
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(METADATA_FILE):
        return False
    try:
        with open(METADATA_FILE, 'r') as f:
            meta = json.load(f)
        return (
            meta.get('model') == MODEL_NAME and
            meta.get('text_format') == TEXT_FORMAT_VERSION and
            meta.get('num_products') == num_products
        )
    except Exception:
        return False


def _save_metadata(num_products):
    meta = {
        'model': MODEL_NAME,
        'text_format': TEXT_FORMAT_VERSION,
        'num_products': num_products,
        'device_used': DEVICE,
        'embedding_dim': model.get_sentence_embedding_dimension(),
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(meta, f, indent=2)


@st.cache_resource(show_spinner="Loading model and product data...")
def load_and_preprocess_data():
    if not os.path.exists(PRODUCTS_FILE):
        st.error(f"Dataset file '{PRODUCTS_FILE}' not found! If using Lite mode, please run create_lite_dataset.py first.")
        st.stop()
        
    df_products = pd.read_csv(PRODUCTS_FILE)
    df_categories = pd.read_csv('amazon_categories.csv')
    df_products = _preprocess(df_products, df_categories)
    embeddings = _load_or_create_embeddings(df_products)
    return model, df_products, embeddings


def _preprocess(df_products, df_categories):
    df_products = df_products.merge(df_categories, left_on='category_id', right_on='id', how='left')
    df_products = df_products[[
        'asin', 'title', 'imgUrl', 'productURL',
        'stars', 'reviews', 'price', 'listPrice',
        'category_name', 'isBestSeller', 'boughtInLastMonth'
    ]]
    df_products['title'] = df_products['title'].fillna('No Title')
    df_products['imgUrl'] = df_products['imgUrl'].fillna('')
    df_products['stars'] = df_products['stars'].fillna(0)
    df_products['reviews'] = df_products['reviews'].fillna(0)
    df_products['price'] = df_products['price'].fillna(0)
    df_products['category_name'] = df_products['category_name'].fillna('Uncategorized')
    df_products['boughtInLastMonth'] = df_products['boughtInLastMonth'].fillna(0)
    return df_products


def _load_or_create_embeddings(df_products):
    n = len(df_products)

    if _embeddings_valid(n):
        print(f"[Embeddings] Loading cached embeddings ({n:,} products, model={MODEL_NAME})")
        return np.load(EMBEDDINGS_FILE)

    # Embeddings are stale or missing — regenerate
    print(f"[Embeddings] Generating embeddings for {n:,} products on {DEVICE.upper()}")
    print(f"[Embeddings] Model: {MODEL_NAME}, batch_size={BATCH_SIZE}")

    texts = df_products.apply(_make_product_text, axis=1).tolist()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        device=DEVICE,
        normalize_embeddings=True,   # normalize for faster dot-product similarity
        convert_to_numpy=True,
    )

    np.save(EMBEDDINGS_FILE, embeddings)
    _save_metadata(n)
    print(f"[Embeddings] Saved to {EMBEDDINGS_FILE}")
    return embeddings
