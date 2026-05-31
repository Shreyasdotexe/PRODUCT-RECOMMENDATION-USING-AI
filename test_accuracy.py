# test_accuracy.py

from accuracy_metrics import evaluate_model_accuracy

if __name__ == "__main__":
    print("Starting accuracy evaluation...")
    accuracy = evaluate_model_accuracy()
    print(f"Final Model Accuracy on Test Set: {accuracy * 100:.2f}%")
