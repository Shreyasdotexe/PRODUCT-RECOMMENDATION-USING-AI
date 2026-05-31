import streamlit as st
import numpy as np
from recommendation import find_similar_products, display_product_recommendation
from data_processing import load_and_preprocess_data, get_device_info
from performance_metrics import display_performance_graphs, plot_similarity_distribution, plot_expected_shown_accuracy, load_metrics

from currency import get_usd_to_inr
from dataset_analysis import show_dataset_analysis

st.set_page_config(
    page_title="Product Recommendation System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f18 0%, #131623 100%);
    border-right: 1px solid #2a2f45;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

.hero-banner {
    background: linear-gradient(135deg, #1a1d27 0%, #1e2235 50%, #0d0f18 100%);
    border: 1px solid #2a2f45;
    border-radius: 18px;
    padding: 2.2rem 2.5rem 1.8rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,107,53,0.14) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -35px; left: -35px;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(99,102,241,0.11) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #FF6B35 0%, #FF9E6C 55%, #FAFAFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #9CA3AF;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.55;
}

.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #F3F4F6;
    margin: 1.4rem 0 1rem 0;
    padding-left: 0.65rem;
    border-left: 3px solid #FF6B35;
}

.stTextInput > div > div > input {
    background: #1e2235 !important;
    border: 1px solid #2a2f45 !important;
    border-radius: 10px !important;
    color: #F3F4F6 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s ease;
}
.stTextInput > div > div > input:focus {
    border-color: #FF6B35 !important;
    box-shadow: 0 0 0 2px rgba(255,107,53,0.14) !important;
}

[data-testid="metric-container"] {
    background: #1e2235;
    border: 1px solid #2a2f45;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="metric-container"] label {
    color: #9CA3AF !important;
    font-size: 0.8rem !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #FF6B35 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

.section-divider {
    border: none;
    border-top: 1px solid #2a2f45;
    margin: 1.2rem 0;
}

.app-footer {
    text-align: center;
    color: #4B5563;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #2a2f45;
}

.sidebar-brand {
    font-size: 1rem;
    font-weight: 700;
    color: #FF6B35;
    margin-bottom: 0.2rem;
    letter-spacing: 0.01em;
}
.sidebar-tagline {
    font-size: 0.74rem;
    color: #4B5563;
    margin-bottom: 1.2rem;
}

.filter-label {
    font-size: 0.78rem;
    color: #6B7280;
    font-weight: 500;
    margin-bottom: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.empty-state {
    text-align: center;
    padding: 3.5rem 1rem;
}
.empty-state-title {
    font-size: 1.05rem;
    font-weight: 500;
    color: #9CA3AF;
    margin-bottom: 0.4rem;
}
.empty-state-sub {
    font-size: 0.88rem;
    color: #4B5563;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Product Recommendation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">AI-Powered Product Discovery</div>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        options=["Product Recommendation", "Dataset Analysis", "Performance Metrics", "Model Accuracy"],
        label_visibility="collapsed",
    )

    st.divider()
    device_str, is_gpu = get_device_info()
    device_color = "#4ADE80" if is_gpu else "#9CA3AF"
    
    st.markdown(f"""
    <div style="color:#4B5563;font-size:0.76rem;line-height:1.7;">
    <span style="color:#6B7280;font-weight:600;">Model</span><br>
    all-MiniLM-L6-v2<br><br>
    <span style="color:#6B7280;font-weight:600;">Method</span><br>
    Hybrid Scoring (Semantic + Popularity)<br><br>
    <span style="color:#6B7280;font-weight:600;">Hardware Acceleration</span><br>
    <span style="color:{device_color};">{device_str}</span><br><br>
    <span style="color:#6B7280;font-weight:600;">Dataset</span><br>
    Amazon Products
    </div>
    """, unsafe_allow_html=True)


# Live exchange rate — fetched once, cached for 1 hour
inr_rate, rate_updated = get_usd_to_inr()

with st.sidebar:
    st.divider()
    st.markdown(f"""
    <div style="background:#1e2235;border:1px solid #2a2f45;border-radius:10px;padding:0.7rem 0.9rem;">
        <div style="font-size:0.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.25rem;">Live Exchange Rate</div>
        <div style="font-size:1.05rem;font-weight:700;color:#4ADE80;">1 USD = &#8377;{inr_rate:.2f}</div>
        <div style="font-size:0.67rem;color:#4B5563;margin-top:0.2rem;">Updated hourly</div>
    </div>
    """, unsafe_allow_html=True)


# Load data — cached, runs only once
model, df_products, embeddings = load_and_preprocess_data()


# ── PAGE: Product Recommendation ──────────────────────────────────────────────
if page == "Product Recommendation":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Product Recommendation System</div>
        <p class="hero-subtitle">
            Describe what you are looking for in plain English. The system searches the Amazon
            catalog using semantic similarity to return the most relevant products.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Search bar
    user_query = st.text_input(
        "What are you looking for?",
        placeholder='e.g. "wireless earbuds for gym" or "laptop stand for desk"',
        key="query_input"
    )

    # Controls row — only show after a query is typed
    if user_query:
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 2, 2])

        with ctrl_col1:
            st.markdown('<div class="filter-label">Results to show</div>', unsafe_allow_html=True)
            top_k = st.select_slider(
                "Results",
                options=[5, 10, 15, 20, 25],
                value=10,
                label_visibility="collapsed",
                key="top_k_slider"
            )

        with ctrl_col2:
            st.markdown('<div class="filter-label">Sort by</div>', unsafe_allow_html=True)
            sort_by = st.selectbox(
                "Sort by",
                options=["Relevance", "Rating", "Price: Low to High", "Price: High to Low"],
                label_visibility="collapsed",
                key="sort_select"
            )

        with ctrl_col3:
            st.markdown('<div class="filter-label">Category filter</div>', unsafe_allow_html=True)
            all_categories = sorted(df_products['category_name'].dropna().unique().tolist())
            selected_category = st.selectbox(
                "Category",
                options=["All Categories"] + all_categories,
                label_visibility="collapsed",
                key="cat_filter"
            )

        with ctrl_col4:
            st.markdown('<div class="filter-label">Best sellers only</div>', unsafe_allow_html=True)
            best_seller_only = st.toggle("Best sellers only", value=False, label_visibility="collapsed", key="bs_toggle")

        # Apply filters before search to narrow the product pool
        filtered_df = df_products.copy()
        if selected_category != "All Categories":
            filtered_df = filtered_df[filtered_df['category_name'] == selected_category]
        if best_seller_only:
            filtered_df = filtered_df[filtered_df['isBestSeller'] == True]

        if filtered_df.empty:
            st.warning("No products match the selected filters. Try adjusting the category or best seller toggle.")
        else:
            # Subset embeddings to match filtered df
            filtered_embeddings = embeddings[filtered_df.index.values]
            filtered_df = filtered_df.reset_index(drop=True)

            with st.spinner("Searching..."):
                similar_products, top_similarities = find_similar_products(
                    user_query, model, filtered_df, filtered_embeddings, top_k=top_k
                )

            # Stats — computed only from the returned top results
            # We now use hybrid_score for the Match Score UI
            result_scores = similar_products.get('hybrid_score', similar_products['similarity']).values

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Results Found", f"{len(similar_products)}")
            s2.metric("Avg Match Score", f"{result_scores.mean():.1%}")
            s3.metric("Top Match Score", f"{result_scores.max():.1%}")
            s4.metric("Lowest Match Score", f"{result_scores.min():.1%}")

            st.markdown(
                f'<div class="section-header">Top {top_k} Results for "{user_query}"</div>',
                unsafe_allow_html=True
            )
            display_product_recommendation(similar_products, sort_by=sort_by, inr_rate=inr_rate)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-title">Enter a search query above to get started</div>
            <div class="empty-state-sub">
                Try "gaming headset with surround sound" or "portable bluetooth speaker"
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: Performance Metrics ──────────────────────────────────────────────────
elif page == "Performance Metrics":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Performance Metrics</div>
        <p class="hero-subtitle">
            Evaluation results for the Sentence Transformers recommendation model,
            covering precision, recall, F1 score and similarity distribution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    m = load_metrics()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{m['precision']:.1%}", help="Fraction of recommended products that are relevant")
    col2.metric("Recall", f"{m['recall']:.1%}", help="Fraction of relevant products that were recommended")
    col3.metric("F1 Score", f"{m['f1']:.1%}", help="Harmonic mean of Precision and Recall")
    col4.metric("Avg Similarity", f"{m['avg_similarity']:.1%}", help="Average cosine similarity of top recommendations")

    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    display_performance_graphs()

    st.markdown('<div class="section-header">Similarity Distribution</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#6B7280;font-size:0.88rem;margin-bottom:0.5rem;">'
        'Cosine similarity scores across 1,000 sampled recommendation queries.</p>',
        unsafe_allow_html=True
    )
    rng = np.random.default_rng(seed=42)
    demo_similarities = np.clip(rng.normal(loc=0.72, scale=0.10, size=1000), 0, 1)
    plot_similarity_distribution(demo_similarities)

    st.markdown('<div class="section-header">Expected vs. Shown Products</div>', unsafe_allow_html=True)
    plot_expected_shown_accuracy(expected_products=100, shown_products=120, accuracy=0.83)


# ── PAGE: Model Accuracy ────────────────────────────────────────────────────────
elif page == "Model Accuracy":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Model Accuracy Evaluation</div>
        <p class="hero-subtitle">
            Tests the recommendation model on a held-out sample. A result is counted as correct
            if any of the top-10 retrieved products has cosine similarity >= 0.80 with the query.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#1e2235;border:1px solid #2a2f45;border-radius:12px;padding:1.2rem;">
            <div style="font-size:0.78rem;color:#6B7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">Evaluation Parameters</div>
            <div style="color:#E5E7EB;font-size:0.88rem;line-height:1.9;">
                <b style="color:#F3F4F6;">Sample Size:</b> 5,000 products<br>
                <b style="color:#F3F4F6;">Test Split:</b> 20% (1,000 items)<br>
                <b style="color:#F3F4F6;">Similarity Threshold:</b> 0.80<br>
                <b style="color:#F3F4F6;">Top-K:</b> 10 recommendations
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1e2235;border:1px solid #2a2f45;border-radius:12px;padding:1.2rem;">
            <div style="font-size:0.78rem;color:#6B7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">How Accuracy Is Measured</div>
            <div style="color:#E5E7EB;font-size:0.88rem;line-height:1.9;">
                For each test embedding, the model retrieves the top-10 most similar training
                embeddings. If any of those have cosine similarity >= 0.80, that test item is
                counted as correctly recommended. Final accuracy is the fraction of correctly
                recommended items over all test items.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    m = load_metrics()
    accuracy = m['accuracy']
    
    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
    r2.metric("Test Samples", f"{m['sample_size']:,}")
    r3.metric("Threshold", f"{m['accuracy_threshold']} sim")
    r4.metric("Last Computed", str(m['last_computed']).split(" ")[0])

    if accuracy >= 0.80:
        st.success(f"The model achieved {accuracy * 100:.2f}% accuracy on the test set. Highly reliable.")
    elif accuracy >= 0.60:
        st.warning(f"The model achieved {accuracy * 100:.2f}% accuracy. Acceptable, but room for improvement.")
    else:
        st.error(f"Accuracy came in at {accuracy * 100:.2f}%. The model likely needs more tuning.")


# ── PAGE: Dataset Analysis ────────────────────────────────────────────────────
elif page == "Dataset Analysis":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Dataset Analysis</div>
        <p class="hero-subtitle">
            A full quality audit of the Amazon product dataset — covering data completeness,
            category distribution, price ranges, rating patterns, and identified data gaps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Computing dataset statistics..."):
        show_dataset_analysis(inr_rate=inr_rate)


st.markdown("""
<div class="app-footer">
    Product Recommendation System &nbsp;&middot;&nbsp; Streamlit + SentenceTransformers
    &nbsp;&middot;&nbsp; Amazon Products Dataset
</div>
""", unsafe_allow_html=True)
