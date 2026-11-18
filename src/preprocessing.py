"""
preprocessing.py - Text Preprocessing Module
=============================================
This module handles all NLP text preprocessing steps:
1. Lowercasing
2. Special character removal
3. Extra whitespace removal
4. Tokenization (NLTK)
5. Stopword removal (NLTK)
6. Lemmatization (NLTK WordNetLemmatizer)

This same preprocessing is used during both training and prediction
to ensure consistency.
"""

import os
import re
import nltk

# Ensure local project nltk_data is prioritized to prevent missing resource errors
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_nltk_dir = os.path.join(project_root, 'nltk_data')
if os.path.exists(local_nltk_dir) and local_nltk_dir not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk_dir)

# Ensure resources are available without noisy network attempts if already present
for resource_path, package_name in [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
]:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        try:
            nltk.download(package_name, quiet=True)
        except Exception:
            pass

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Apply the full NLP preprocessing pipeline to a single text string.

    Steps:
        1. Convert to lowercase
        2. Remove special characters (keep only letters and spaces)
        3. Remove extra whitespace
        4. Tokenize into words
        5. Remove stopwords
        6. Lemmatize each token

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned and preprocessed text

    Example:
        >>> preprocess_text("My PAYMENT has FAILED!!!")
        'payment fail'
    """
    # Handle missing or non-string values
    if not isinstance(text, str):
        return ""

    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Remove special characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Step 3: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 4: Tokenization using NLTK
    tokens = word_tokenize(text)

    # Step 5: Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Step 6: Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a single string
    processed_text = ' '.join(tokens)

    return processed_text


def preprocess_dataframe(df, text_column='Ticket Description'):
    """
    Apply preprocessing to an entire DataFrame column.

    Args:
        df (pandas.DataFrame): DataFrame containing text data
        text_column (str): Name of the column containing text to preprocess

    Returns:
        pandas.DataFrame: DataFrame with an added 'cleaned_text' column
    """
    df = df.copy()
    print(f"Preprocessing {len(df)} rows in column '{text_column}'...")
    df['cleaned_text'] = df[text_column].apply(preprocess_text)
    return df



if __name__ == "__main__":
    # Quick test of the preprocessing pipeline
    test_queries = [
        "My PAYMENT has FAILED!!!",
        "Where is my order???",
        "I want my money back!!!",
        "The product I received is DAMAGED...",
        "I cannot login to my account",
    ]

    print("=" * 60)
    print("TEXT PREPROCESSING DEMONSTRATION")
    print("=" * 60)

    for query in test_queries:
        cleaned = preprocess_text(query)
        print(f"\nOriginal : {query}")
        print(f"Processed: {cleaned}")
