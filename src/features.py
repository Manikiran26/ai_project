"""
features.py - Feature Extraction Module
========================================
This module implements two feature extraction methods for converting
preprocessed text into numerical feature vectors:

1. Bag of Words (BoW) - using CountVectorizer
2. TF-IDF - using TfidfVectorizer

IMPORTANT: Vectorizers are fitted ONLY on training data to prevent
data leakage. Test data is transformed using the already-fitted vectorizer.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_bow_features(X_train, X_test):
    """
    Extract Bag of Words features from text data.

    Bag of Words (BoW) represents each document as a vector of word counts.
    Each unique word in the corpus becomes a feature, and the value is the
    count of how many times that word appears in the document.

    IMPORTANT: The vectorizer is fit ONLY on X_train to avoid data leakage.

    Args:
        X_train (array-like): Training text data (preprocessed)
        X_test (array-like): Testing text data (preprocessed)

    Returns:
        tuple: (X_train_bow, X_test_bow, bow_vectorizer)
            - X_train_bow: Sparse matrix of training features
            - X_test_bow: Sparse matrix of testing features
            - bow_vectorizer: Fitted CountVectorizer object
    """
    # Initialize CountVectorizer for Bag of Words
    bow_vectorizer = CountVectorizer()

    # Fit on training data ONLY, then transform training data
    X_train_bow = bow_vectorizer.fit_transform(X_train)

    # Transform test data using the already-fitted vectorizer
    X_test_bow = bow_vectorizer.transform(X_test)

    print(f"[BoW] Vocabulary size: {len(bow_vectorizer.vocabulary_)}")
    print(f"[BoW] Training matrix shape: {X_train_bow.shape}")
    print(f"[BoW] Testing matrix shape: {X_test_bow.shape}")

    return X_train_bow, X_test_bow, bow_vectorizer


def get_tfidf_features(X_train, X_test):
    """
    Extract TF-IDF features from text data.

    TF-IDF (Term Frequency - Inverse Document Frequency) represents each
    document as a vector where each word's value reflects how important
    it is to that document relative to the entire corpus.

    - TF (Term Frequency): How often a word appears in a document
    - IDF (Inverse Document Frequency): How rare a word is across all documents
    - TF-IDF = TF x IDF

    Words that are common across all documents get lower weights,
    while words that are distinctive to specific documents get higher weights.

    IMPORTANT: The vectorizer is fit ONLY on X_train to avoid data leakage.

    Args:
        X_train (array-like): Training text data (preprocessed)
        X_test (array-like): Testing text data (preprocessed)

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, tfidf_vectorizer)
            - X_train_tfidf: Sparse matrix of training features
            - X_test_tfidf: Sparse matrix of testing features
            - tfidf_vectorizer: Fitted TfidfVectorizer object
    """
    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit on training data ONLY, then transform training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Transform test data using the already-fitted vectorizer
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"[TF-IDF] Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"[TF-IDF] Training matrix shape: {X_train_tfidf.shape}")
    print(f"[TF-IDF] Testing matrix shape: {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer


if __name__ == "__main__":
    # Quick demonstration of feature extraction
    sample_train = [
        "payment fail money deducted",
        "order arrive delivery late",
        "want refund money back",
        "cancel order subscription",
    ]
    sample_test = [
        "payment issue charged twice",
        "delivery delayed package",
    ]

    print("=" * 60)
    print("FEATURE EXTRACTION DEMONSTRATION")
    print("=" * 60)

    print("\n--- Bag of Words ---")
    X_tr_bow, X_te_bow, bow_vec = get_bow_features(sample_train, sample_test)
    print(f"BoW feature names (first 10): {bow_vec.get_feature_names_out()[:10]}")

    print("\n--- TF-IDF ---")
    X_tr_tfidf, X_te_tfidf, tfidf_vec = get_tfidf_features(sample_train, sample_test)
    print(f"TF-IDF feature names (first 10): {tfidf_vec.get_feature_names_out()[:10]}")
