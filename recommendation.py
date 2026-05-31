import pandas as pd
import streamlit as st
import numpy as np
import html as html_lib
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import load_and_preprocess_data
from currency import format_inr


def find_similar_products(user_query, model, df_products, embeddings, top_k=10):
    query_embedding = model.encode([user_query])[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Instead of ranking 1.4 million products by hybrid score (slow),
    # first get the top 500 candidates by pure semantic similarity.
    candidate_indices = np.argsort(similarities)[-500:][::-1]
    candidates = df_products.iloc[candidate_indices].copy()
    candidates['similarity'] = similarities[candidate_indices]

    # Hybrid Scoring Components
    # 1. Semantic Similarity (0 to 1) -> 70% weight
    
    # 2. Normalized Stars (0 to 1) -> 15% weight
    norm_stars = candidates['stars'] / 5.0
    
    # 3. Normalized Reviews (log scale, cap at 5000) -> 5% weight
    norm_reviews = np.clip(np.log1p(candidates['reviews'].astype(float)) / np.log1p(5000.0), 0, 1)
    
    # 4. Best Seller Bonus (0 or 1) -> 5% weight
    is_best_seller = candidates['isBestSeller'].astype(float)
    
    # 5. Recent Purchase Bonus (log scale, cap at 5000) -> 5% weight
    norm_bought = np.clip(np.log1p(candidates['boughtInLastMonth'].astype(float)) / np.log1p(5000.0), 0, 1)
    
    candidates['hybrid_score'] = (
        (candidates['similarity'] * 0.70) +
        (norm_stars * 0.15) +
        (norm_reviews * 0.05) +
        (is_best_seller * 0.05) +
        (norm_bought * 0.05)
    )

    top_products = candidates.nlargest(top_k, 'hybrid_score')
    
    # We still return the full similarities array so stats can be computed over the whole dataset if needed,
    # but the app uses it for the top returned results.
    return top_products, similarities


def _render_stars(rating):
    try:
        rating = float(rating)
    except Exception:
        rating = 0.0
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    stars_html = "&#9733;" * full + ("&#189;" if half else "") + "&#9734;" * empty
    return f'<span style="color:#FFB800;font-size:0.95rem;">{stars_html}</span> <span style="color:#9CA3AF;font-size:0.82rem;">{rating:.1f}</span>'


def _format_reviews(reviews):
    try:
        r = int(reviews)
        if r == 0:
            return "No reviews yet"
        if r >= 1000:
            return f"{r / 1000:.1f}k reviews"
        return f"{r} reviews"
    except Exception:
        return "No reviews yet"


def _inject_card_styles():
    st.markdown("""
    <style>
    .product-card {
        background: linear-gradient(145deg, #1e2235, #181b2a);
        border: 1px solid #2a2f45;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1.1rem;
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        position: relative;
        overflow: hidden;
    }
    .product-card:hover {
        transform: translateY(-3px);
        border-color: #FF6B35;
        box-shadow: 0 6px 28px rgba(255, 107, 53, 0.16);
    }
    .product-img-wrap {
        width: 100%;
        height: 170px;
        background: #252838;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    .product-img-wrap img {
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
    }
    .product-img-placeholder {
        color: #4B5563;
        font-size: 0.82rem;
    }
    .product-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #F3F4F6;
        margin: 0 0 0.35rem 0;
        line-height: 1.35;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 2.4rem;
    }
    .product-price {
        font-size: 1.15rem;
        font-weight: 700;
        color: #4ADE80;
        margin: 0.35rem 0;
    }
    .product-reviews {
        font-size: 0.78rem;
        color: #6B7280;
        margin: 0.15rem 0 0.45rem 0;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin: 0.4rem 0 0.55rem 0;
    }
    .badge {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-category {
        background: rgba(99,102,241,0.18);
        color: #a5b4fc;
        border: 1px solid rgba(99,102,241,0.3);
    }
    .badge-bestseller {
        background: rgba(255,107,53,0.15);
        color: #FB923C;
        border: 1px solid rgba(255,107,53,0.35);
    }
    .match-label {
        font-size: 0.74rem;
        color: #6B7280;
        margin: 0.5rem 0 0.2rem 0;
        display: flex;
        justify-content: space-between;
    }
    .match-bar-bg {
        background: #252838;
        border-radius: 6px;
        height: 6px;
        overflow: hidden;
        margin-bottom: 0.7rem;
    }
    .match-bar-fill {
        height: 100%;
        border-radius: 6px;
    }
    .view-btn {
        display: block;
        text-align: center;
        padding: 7px 0;
        background: linear-gradient(135deg, #FF6B35, #d95a26);
        color: #fff !important;
        border-radius: 8px;
        text-decoration: none !important;
        font-size: 0.82rem;
        font-weight: 600;
        transition: opacity 0.18s;
        margin-top: 0.3rem;
    }
    .view-btn:hover { opacity: 0.84; }
    </style>
    """, unsafe_allow_html=True)


def _build_card_html(row, inr_rate=84.0):
    title_raw = str(row.get('title', 'Unknown Product'))
    title_safe = html_lib.escape(title_raw)

    category_raw = str(row.get('category_name', 'Uncategorized'))
    category_safe = html_lib.escape(category_raw[:28])

    price = row.get('price', 0)
    price_display = format_inr(price, inr_rate)

    stars = row.get('stars', 0)
    reviews = row.get('reviews', 0)
    is_best_seller = row.get('isBestSeller', False)
    img_url = str(row.get('imgUrl', ''))
    product_url = html_lib.escape(str(row.get('productURL', '#')))
    
    # Use hybrid score if available, fallback to pure similarity
    match_score_val = float(row.get('hybrid_score', row.get('similarity', 0)))
    sim_pct = max(0, min(100, int(match_score_val * 100)))

    if img_url and img_url.startswith('http'):
        img_block = f'<img src="{html_lib.escape(img_url)}" alt="{title_safe}" style="max-height:100%;max-width:100%;object-fit:contain;" />'
    else:
        img_block = '<span class="product-img-placeholder">No image</span>'

    badges = f'<span class="badge badge-category">{category_safe}</span>'
    if is_best_seller:
        badges += '<span class="badge badge-bestseller">Best Seller</span>'

    if sim_pct >= 70:
        bar_color = "#4ADE80"
    elif sim_pct >= 50:
        bar_color = "#FFB800"
    else:
        bar_color = "#FF6B35"

    stars_html = _render_stars(stars)
    reviews_text = _format_reviews(reviews)

    card = f"""
<div class="product-card">
  <div class="product-img-wrap">{img_block}</div>
  <div class="product-title" title="{title_safe}">{title_safe}</div>
  <div>{stars_html}</div>
  <div class="product-reviews">{reviews_text}</div>
  <div class="product-price">{price_display}</div>
  <div class="badge-row">{badges}</div>
  <div class="match-label">
    <span>Match Score</span>
    <span style="color:#F3F4F6;font-weight:600;">{sim_pct}%</span>
  </div>
  <div class="match-bar-bg">
    <div class="match-bar-fill" style="width:{sim_pct}%;background:{bar_color};"></div>
  </div>
  <a class="view-btn" href="{product_url}" target="_blank">View on Amazon</a>
</div>"""
    return card


def display_product_recommendation(similar_products, sort_by="Relevance", inr_rate=84.0):
    if not isinstance(similar_products, pd.DataFrame) or similar_products.empty:
        st.error("No recommendations found. Try a different search term.")
        return

    # Apply sorting
    df = similar_products.copy()
    if sort_by == "Rating":
        df = df.sort_values('stars', ascending=False)
    elif sort_by == "Price: Low to High":
        df['_price_num'] = pd.to_numeric(
            df['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce'
        )
        df = df.sort_values('_price_num', ascending=True)
    elif sort_by == "Price: High to Low":
        df['_price_num'] = pd.to_numeric(
            df['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce'
        )
        df = df.sort_values('_price_num', ascending=False)
    # Default: keep similarity order

    _inject_card_styles()

    rows = list(df.iterrows())
    col_left, col_right = st.columns(2)
    inr_rate_local = inr_rate

    for i, (_, row) in enumerate(rows):
        card_html = _build_card_html(row, inr_rate_local)
        if i % 2 == 0:
            with col_left:
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            with col_right:
                st.markdown(card_html, unsafe_allow_html=True)
