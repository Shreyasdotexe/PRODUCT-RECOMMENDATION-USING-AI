import pandas as pd
import numpy as np

def create_lite_dataset(target_size=50000):
    print(f"Creating a {target_size}-product Lite dataset for free deployment...")
    
    # 1. Load full data
    print("Loading full dataset...")
    df_full = pd.read_csv('amazon_products.csv')
    df_cat = pd.read_csv('amazon_categories.csv')
    embeddings_full = np.load('product_embeddings.npy')
    
    # Merge category names so we can stratify by them
    df_full = df_full.merge(df_cat, left_on='category_id', right_on='id', how='left')
    df_full['category_name'] = df_full['category_name'].fillna('Uncategorized')
    
    # 2. Score products to keep the "best" ones
    print("Ranking products by popularity and rating...")
    df_full['popularity_score'] = (
        (df_full['stars'].fillna(0) / 5.0) * 0.4 + 
        (np.clip(df_full['reviews'].fillna(0) / 5000, 0, 1)) * 0.4 +
        (df_full['isBestSeller'].fillna(False).astype(int)) * 0.2
    )
    
    # 3. Stratified sampling to ensure ALL categories are included
    print("Ensuring category coverage...")
    unique_categories = df_full['category_name'].unique()
    num_categories = len(unique_categories)
    
    # Give every category a baseline of top N products (e.g., 50)
    baseline_per_cat = 50
    lite_indices = []
    
    for cat in unique_categories:
        cat_items = df_full[df_full['category_name'] == cat]
        # Take the top N from this category
        top_cat_items = cat_items.nlargest(baseline_per_cat, 'popularity_score')
        lite_indices.extend(top_cat_items.index.tolist())
    
    lite_indices = list(set(lite_indices)) # remove duplicates if any
    
    # Fill the remaining slots up to target_size with the globally most popular products
    remaining_slots = target_size - len(lite_indices)
    
    if remaining_slots > 0:
        print(f"Filled {len(lite_indices)} slots via category baselines. Filling remaining {remaining_slots} globally...")
        # Get products not already in lite_indices
        remaining_df = df_full.drop(lite_indices)
        top_global = remaining_df.nlargest(remaining_slots, 'popularity_score')
        lite_indices.extend(top_global.index.tolist())
    elif remaining_slots < 0:
        # If we somehow exceeded target_size (unlikely with 50 per cat for ~250 cats), truncate
        print("Category baseline exceeded target size. Truncating...")
        lite_indices = lite_indices[:target_size]
        
    # 4. Slice data
    print("Slicing data...")
    # We must slice the original CSV before the merge so it matches the original format perfectly
    df_original = pd.read_csv('amazon_products.csv')
    df_lite = df_original.iloc[lite_indices].copy()
    embeddings_lite = embeddings_full[lite_indices]
    
    # 5. Save Lite files
    print("Saving lite files...")
    df_lite.to_csv('amazon_products_lite.csv', index=False)
    np.save('product_embeddings_lite.npy', embeddings_lite)
    
    print(f"Done! Created:")
    print(f"- amazon_products_lite.csv (Rows: {len(df_lite)})")
    print(f"- product_embeddings_lite.npy")
    print("\nThese files are completely separate from your main dataset.")

if __name__ == "__main__":
    create_lite_dataset()
