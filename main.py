"""
main.py - Complete Training and Evaluation Pipeline
====================================================
Run this script to execute the entire ML pipeline:

    python main.py

This will:
    1. Load and explore the dataset
    2. Clean and preprocess text data
    3. Perform train/test split
    4. Extract BoW and TF-IDF features
    5. Train 4 ML models on both feature sets (8 combinations)
    6. Evaluate all models
    7. Generate comparison plots and confusion matrices
    8. Select the best model
    9. Save the final model and vectorizer
    10. Test predictions on sample queries
"""

import os
import sys
import joblib

# Ensure the project root is in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.train import load_and_prepare_data, train_all_models
from src.evaluate import (
    evaluate_all_models,
    find_best_model,
    plot_model_comparison,
    plot_class_distribution,
)
from src.predict import predict_and_respond


def main():
    """Run the complete ML pipeline."""

    print("+" + "=" * 58 + "+")
    print("|  AI-Based Customer Support Query Classification System  |")
    print("|            Complete Training Pipeline                   |")
    print("+" + "=" * 58 + "+")

    # ----- Paths -----
    data_path = os.path.join(project_root, 'data', 'customer_support_tickets.csv')
    models_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root, 'results')

    # Ensure directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ==========================================
    # STEP 1: Load, Explore, and Preprocess Data
    # ==========================================
    X_train, X_test, y_train, y_test, df = load_and_prepare_data(data_path)

    # Plot class distribution
    plot_class_distribution(df, results_dir)

    # ==========================================
    # STEP 2: Train All Models (BoW + TF-IDF)
    # ==========================================
    results, bow_vectorizer, tfidf_vectorizer = train_all_models(
        X_train, X_test, y_train, y_test, models_dir
    )

    # ==========================================
    # STEP 3: Evaluate All Models
    # ==========================================
    comparison_df = evaluate_all_models(results, results_dir)

    # ==========================================
    # STEP 4: Generate Comparison Plots
    # ==========================================
    plot_model_comparison(comparison_df, results_dir)

    # ==========================================
    # STEP 5: Select Best Model
    # ==========================================
    best_feature, best_model_name, best_f1 = find_best_model(comparison_df)

    # ==========================================
    # STEP 6: Save Final Model and Vectorizer
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 6: SAVING FINAL MODEL")
    print("=" * 60)

    # Determine the key for the best model in results
    if best_feature == 'BoW':
        best_key = f"BoW_{best_model_name}"
        best_vectorizer = bow_vectorizer
    else:
        best_key = f"TFIDF_{best_model_name}"
        best_vectorizer = tfidf_vectorizer

    best_model_obj = results[best_key]['model']
    best_row = comparison_df[(comparison_df['Model'] == best_model_name) & (comparison_df['Feature Method'] == best_feature)].iloc[0]

    # Save final model
    final_model_path = os.path.join(models_dir, 'final_model.pkl')
    joblib.dump(best_model_obj, final_model_path)
    print(f"  Final model saved: {final_model_path}")

    # Save final vectorizer
    final_vectorizer_path = os.path.join(models_dir, 'final_vectorizer.pkl')
    joblib.dump(best_vectorizer, final_vectorizer_path)
    print(f"  Final vectorizer saved: {final_vectorizer_path}")

    # Save model info
    model_info = {
        'model_name': best_model_name,
        'feature_method': best_feature,
        'accuracy': float(best_row['Accuracy']),
        'f1_score': float(best_f1),
        'precision': float(best_row['Precision']),
        'recall': float(best_row['Recall']),
        'dataset_size': int(len(df)),
        'categories': sorted(y_train.unique().tolist()),
        'num_categories': int(len(y_train.unique())),
    }
    info_path = os.path.join(models_dir, 'model_info.pkl')
    joblib.dump(model_info, info_path)
    print(f"  Model info saved: {info_path}")

    # ==========================================
    # STEP 7: Test Predictions (Unseen Queries)
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 7: TESTING PREDICTIONS ON 12 NEW QUERIES")
    print("=" * 60)

    test_queries = [
        "I am having trouble with my payment transaction and deduction.",
        "I need help getting a refund for my recent purchase.",
        "I cannot access my account because password reset email is not arriving.",
        "The delivery package has not arrived yet despite being overdue.",
        "The device battery life drains extremely quickly after charging.",
        "There is a severe software bug crashing the desktop program.",
        "Can you recommend a suitable product for graphic design tasks?",
        "I need technical assistance with initial product setup and configuration.",
        "The display screen is flickering and showing black horizontal lines.",
        "I am experiencing a network connectivity and Wi-Fi disconnect problem.",
        "I lost all my personal data and document backups unexpectedly.",
        "I would like to cancel my order before shipment proceeds.",
    ]

    for query in test_queries:
        result = predict_and_respond(query, best_model_obj, best_vectorizer)
        print(f"\n  Query:    \"{result['original_query']}\"")
        print(f"  Cleaned:  \"{result['cleaned_query']}\"")
        print(f"  Category: {result['predicted_category']}")
        if result['confidence'] is not None:
            print(f"  Confidence: {result['confidence']:.2%}")
        else:
            print("  Confidence: Prediction generated by the trained classification model.")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "+" + "=" * 58 + "+")
    print("|                   PIPELINE COMPLETE                     |")
    print("+" + "=" * 58 + "+")
    print(f"""
Summary:
  Dataset:              {df.shape[0]} samples, {model_info['num_categories']} categories
  NLP Preprocessing:    Lowercasing, Special Char Removal, Whitespace Normalization,
                        NLTK Tokenization, Stopword Removal, WordNet Lemmatization
  Feature Extraction:   Bag of Words (CountVectorizer) and TF-IDF (TfidfVectorizer)
  ML Algorithms:        Logistic Regression, Naive Bayes, Linear SVM, Random Forest
  Best Model:           {best_model_name}
  Best Feature Method:  {best_feature}
  Best F1 Score:        {best_f1}
  Best Accuracy:        {best_row['Accuracy']}

Files Generated:
  - data/processed/customer_support_processed.csv
  - models/final_model.pkl
  - models/final_vectorizer.pkl
  - models/model_info.pkl
  - results/model_comparison.csv
  - results/confusion_matrices/ (8 confusion matrix plots)
  - results/plots/ (comparison charts)

To launch the web app:
  streamlit run app/app.py
""")


if __name__ == "__main__":
    main()

