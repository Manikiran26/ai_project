"""
train.py - Model Training Module
=================================
This module trains four ML classifiers on both BoW and TF-IDF features:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Support Vector Machine (LinearSVC)
4. Random Forest

Each model is trained on both feature representations (BoW and TF-IDF)
to allow comparison of which combination works best.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Add project root to path so we can import our modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_dataframe
from src.features import get_bow_features, get_tfidf_features


def load_and_prepare_data(data_path):
    """
    Load dataset, explore structure, clean data, preprocess text, and perform train/test split.

    Args:
        data_path (str): Path to the CSV dataset

    Returns:
        tuple: (X_train, X_test, y_train, y_test, df)
    """
    print("=" * 60)
    print("STEP 1: LOADING, EXPLORING, AND PREPARING DATA")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(data_path)
    print(f"\nInitial Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Column names:", df.columns.tolist())

    # Exploration & Diagnostics
    missing_desc = df['Ticket Description'].isnull().sum()
    missing_subj = df['Ticket Subject'].isnull().sum()
    print(f"\nMissing 'Ticket Description': {missing_desc}")
    print(f"Missing 'Ticket Subject': {missing_subj}")

    # Remove rows where Ticket Description or Ticket Subject is missing
    initial_len = len(df)
    df = df.dropna(subset=['Ticket Description', 'Ticket Subject']).copy()
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with missing description or subject.")

    # Duplicate records check
    duplicate_count = df.duplicated(subset=['Ticket Description']).sum()
    print(f"Duplicate 'Ticket Description' records: {duplicate_count}")
    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate description rows to prevent data leakage...")
        df = df.drop_duplicates(subset=['Ticket Description']).reset_index(drop=True)

    unique_classes = df['Ticket Subject'].nunique()
    print(f"\nNumber of unique Ticket Subject classes: {unique_classes}")
    print("\nClass Distribution:")
    print(df['Ticket Subject'].value_counts())

    # Check if preprocessed file already exists to avoid redundant slow tokenization
    processed_dir = os.path.join(os.path.dirname(data_path), 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, 'customer_support_processed.csv')

    if os.path.exists(processed_path):
        print(f"\nLoading existing preprocessed dataset from: {processed_path}")
        df_proc = pd.read_csv(processed_path)
        if len(df_proc) == len(df) and 'cleaned_text' in df_proc.columns:
            df = df_proc
            print("Loaded preprocessed text from cache.")
        else:
            print("Applying NLP preprocessing (lower, regex, tokenize, stopwords, lemmatize)...")
            df = preprocess_dataframe(df, text_column='Ticket Description')
            df.to_csv(processed_path, index=False)
            print(f"Cleaned dataset saved to: {processed_path}")
    else:
        print("\nApplying NLP preprocessing (lower, regex, tokenize, stopwords, lemmatize)...")
        df = preprocess_dataframe(df, text_column='Ticket Description')
        df.to_csv(processed_path, index=False)
        print(f"Cleaned dataset saved to: {processed_path}")

    # Train/Test Split (80/20 with stratification, random_state=42)
    # Features: strictly cleaned 'Ticket Description' (no post-resolution info)
    # Target: 'Ticket Subject'
    X = df['cleaned_text'].fillna('')
    y = df['Ticket Subject']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"\nTrain / Test Split (80/20 stratified, random_state=42):")
    print(f"  Training samples: {len(X_train)} (80%)")
    print(f"  Testing samples:  {len(X_test)} (20%)")
    print(f"  Target classes:   {y.nunique()}")

    return X_train, X_test, y_train, y_test, df


def get_models():
    """
    Return a dictionary of ML models to train.

    Returns:
        dict: Model name -> model instance
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(
            dual='auto',
            max_iter=2000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
    }
    return models


def train_all_models(X_train, X_test, y_train, y_test, models_dir):
    """
    Train all models on both BoW and TF-IDF features.

    Args:
        X_train: Training text data (preprocessed strings)
        X_test: Testing text data (preprocessed strings)
        y_train: Training labels
        y_test: Testing labels
        models_dir (str): Directory to save trained models

    Returns:
        dict: Results dictionary with trained models and predictions
    """
    os.makedirs(models_dir, exist_ok=True)

    results = {}

    # ===== EXPERIMENT A: Bag of Words Features =====
    print("\n" + "=" * 60)
    print("STEP 2A: FEATURE EXTRACTION - BAG OF WORDS")
    print("=" * 60)

    X_train_bow, X_test_bow, bow_vectorizer = get_bow_features(X_train, X_test)

    # Save BoW vectorizer
    joblib.dump(bow_vectorizer, os.path.join(models_dir, 'bow_vectorizer.pkl'))

    # Train models on BoW features
    print("\n--- Training Models on BoW Features ---")
    models = get_models()

    for name, model in models.items():
        print(f"\nTraining {name} (BoW)...")
        model.fit(X_train_bow, y_train)
        y_pred = model.predict(X_test_bow)

        key = f"BoW_{name}"
        results[key] = {
            'feature_method': 'BoW',
            'model_name': name,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
        }

        # Save model
        model_filename = f"bow_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, os.path.join(models_dir, model_filename))
        print(f"  Saved: {model_filename}")

    # ===== EXPERIMENT B: TF-IDF Features =====
    print("\n" + "=" * 60)
    print("STEP 2B: FEATURE EXTRACTION - TF-IDF")
    print("=" * 60)

    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = get_tfidf_features(X_train, X_test)

    # Save TF-IDF vectorizer
    joblib.dump(tfidf_vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))

    # Train models on TF-IDF features
    print("\n--- Training Models on TF-IDF Features ---")
    models = get_models()  # Fresh model instances

    for name, model in models.items():
        print(f"\nTraining {name} (TF-IDF)...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        key = f"TFIDF_{name}"
        results[key] = {
            'feature_method': 'TF-IDF',
            'model_name': name,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
        }

        # Save model
        model_filename = f"tfidf_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, os.path.join(models_dir, model_filename))
        print(f"  Saved: {model_filename}")

    print(f"\nAll models trained and saved to: {models_dir}")

    return results, bow_vectorizer, tfidf_vectorizer


if __name__ == "__main__":
    # Run training standalone
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'customer_support_tickets.csv')
    models_dir = os.path.join(project_root, 'models')

    X_train, X_test, y_train, y_test, df = load_and_prepare_data(data_path)
    results, bow_vec, tfidf_vec = train_all_models(
        X_train, X_test, y_train, y_test, models_dir
    )

    print("\n[OK] Training complete!")
    print(f"  Total model combinations: {len(results)}")
