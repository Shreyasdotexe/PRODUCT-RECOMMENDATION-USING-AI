import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import load_and_preprocess_data

def compute_metrics(sample_size=5000, top_k=10, accuracy_threshold=0.80):
    print("Loading data and embeddings...")
    model, df_products, embeddings = load_and_preprocess_data()
    
    total_products = len(df_products)
    sample_size = min(sample_size, total_products)
    
    print(f"Sampling {sample_size} products for evaluation...")
    sampled_df = df_products.sample(n=sample_size, random_state=42)
    test_indices = sampled_df.index.values
    test_embeddings = embeddings[test_indices]
    test_categories = sampled_df['category_name'].values
    
    print("Computing metrics (this may take a minute)...")
    
    total_precision = 0.0
    hits_for_accuracy = 0
    total_similarity = 0.0
    hit_rate = 0.0
    
    # Process in batches to avoid memory overflow on large distance matrices
    batch_size = 500
    for i in range(0, sample_size, batch_size):
        end_idx = min(i + batch_size, sample_size)
        batch_embs = test_embeddings[i:end_idx]
        batch_cats = test_categories[i:end_idx]
        
        # Compute similarities against ALL products
        sim_matrix = cosine_similarity(batch_embs, embeddings)
        
        for j in range(len(batch_embs)):
            sims = sim_matrix[j]
            # Exclude the query product itself by zeroing its similarity
            query_idx = test_indices[i + j]
            sims[query_idx] = -1.0
            
            # Get top K indices
            top_k_idx = np.argsort(sims)[-top_k:][::-1]
            top_k_sims = sims[top_k_idx]
            top_k_cats = df_products.iloc[top_k_idx]['category_name'].values
            
            # 1. Precision@K (Category Match)
            matches = sum(1 for c in top_k_cats if c == batch_cats[j])
            total_precision += (matches / top_k)
            
            # 2. Hit Rate (Recall proxy)
            if matches > 0:
                hit_rate += 1
                
            # 3. Accuracy (Similarity threshold)
            if any(s >= accuracy_threshold for s in top_k_sims):
                hits_for_accuracy += 1
                
            # 4. Avg Similarity
            total_similarity += np.mean(top_k_sims)
            
        print(f"  Processed {end_idx}/{sample_size}")

    # Final calculations
    precision = total_precision / sample_size
    recall = hit_rate / sample_size  # Fraction of queries where we found at least 1 relevant item
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = hits_for_accuracy / sample_size
    avg_sim = total_similarity / sample_size
    
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_similarity": float(avg_sim),
        "accuracy": float(accuracy),
        "sample_size": sample_size,
        "top_k": top_k,
        "accuracy_threshold": accuracy_threshold,
        "last_computed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall:    {metrics['recall']:.1%}")
    print(f"F1 Score:  {metrics['f1']:.1%}")
    print(f"Avg Sim:   {metrics['avg_similarity']:.1%}")
    print(f"Accuracy:  {metrics['accuracy']:.1%}")
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("\nSaved metrics to metrics.json")

if __name__ == "__main__":
    compute_metrics()
