from data_processing import load_and_preprocess_data

if __name__ == "__main__":
    print("Initializing embedding generation process...")
    # This will trigger the regeneration because model name and text format changed
    model, df_products, embeddings = load_and_preprocess_data()
    print("Done! Embeddings are fully upgraded and saved to disk.")
