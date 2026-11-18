# AI-Based Customer Support Query Classification and Response System

An academic AIML project that demonstrates traditional NLP, feature engineering, supervised machine learning, model evaluation, and real-time prediction for classifying customer support queries.

## Project Overview

This system takes a customer support query (e.g., *"My payment failed but money was deducted"*) and automatically:

1. Cleans and preprocesses the text using NLP techniques
2. Converts text into numerical features using **Bag of Words** and **TF-IDF**
3. Classifies the query using multiple ML algorithms
4. Predicts the support category
5. Generates a predefined support response

### Supported Categories

| Category | Example Query |
|---|---|
| Delivery | "Where is my order?" |
| Payment | "My payment failed" |
| Refund | "I want my money back" |
| Cancellation | "I want to cancel my order" |
| Product Issue | "The product I received is damaged" |
| Complaint | "I am unhappy with the service" |
| Account Issue | "I cannot login to my account" |

## Project Pipeline

```
Customer Query → Dataset → Data Cleaning → NLP Preprocessing
→ Train/Test Split → Feature Extraction (BoW + TF-IDF)
→ ML Models (LR, NB, SVM, RF) → Evaluation → Best Model Selection
→ Save Model → Prediction → Category → Support Response
```

## Technologies Used

- **Python** — Core language
- **Pandas / NumPy** — Data manipulation
- **NLTK** — NLP preprocessing (tokenization, stopword removal, lemmatization)
- **Scikit-learn** — Feature extraction, ML models, evaluation
- **Matplotlib / Seaborn** — Visualizations
- **Joblib** — Model serialization
- **Streamlit** — Web interface

## Project Structure

```
aimlproject/
│
├── data/
│   ├── raw/
│   │   └── customer_support.csv          # Dataset
│   └── processed/
│       └── customer_support_processed.csv # After preprocessing
│
├── notebooks/
│   └── experimentation.ipynb             # Step-by-step notebook
│
├── src/
│   ├── preprocessing.py                  # NLP preprocessing functions
│   ├── features.py                       # BoW and TF-IDF feature extraction
│   ├── train.py                          # Model training
│   ├── evaluate.py                       # Evaluation and visualization
│   └── predict.py                        # Prediction and response system
│
├── models/
│   ├── final_model.pkl                   # Best trained model
│   ├── final_vectorizer.pkl              # Fitted vectorizer
│   └── model_info.pkl                    # Model metadata
│
├── results/
│   ├── model_comparison.csv              # Evaluation results table
│   ├── confusion_matrices/               # Confusion matrix plots
│   └── plots/                            # Comparison charts
│
├── app/
│   └── app.py                            # Streamlit web application
│
├── main.py                               # Complete pipeline runner
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Training Pipeline

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Train 4 models × 2 feature methods = 8 combinations
- Evaluate all models with accuracy, precision, recall, F1 score
- Generate confusion matrices and comparison plots
- Select the best model and save it
- Test predictions on sample queries

### 3. Launch the Web Application

```bash
streamlit run app/app.py
```

### 4. (Optional) Run the Notebook

Open `notebooks/experimentation.ipynb` in Jupyter Notebook for a step-by-step walkthrough.

## ML Models Used

| Model | Description |
|---|---|
| Logistic Regression | Linear model for classification using log-odds |
| Multinomial Naive Bayes | Probabilistic model based on Bayes' theorem |
| Support Vector Machine | Finds optimal hyperplane to separate classes |
| Random Forest | Ensemble of decision trees |

## Evaluation Metrics

- **Accuracy** — Overall correctness
- **Precision** — Of predicted positives, how many are correct
- **Recall** — Of actual positives, how many were found
- **F1 Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Visual breakdown of predictions vs truth

## NLP Preprocessing Steps

1. **Lowercasing** — Normalize case
2. **Special Character Removal** — Remove punctuation and symbols
3. **Whitespace Removal** — Clean extra spaces
4. **Tokenization** — Split text into words (NLTK)
5. **Stopword Removal** — Remove common words (the, is, at, etc.)
6. **Lemmatization** — Reduce words to base form (running → run)

## Feature Extraction

- **Bag of Words (BoW)** — Counts word occurrences in each document
- **TF-IDF** — Weighs words by frequency and rarity across documents

Both methods are compared experimentally to determine which works better.

## Academic Concepts Demonstrated

- Natural Language Processing (NLP)
- Text Preprocessing Pipeline
- Feature Engineering for Text Data
- Supervised Machine Learning
- Multiclass Classification
- Model Evaluation and Comparison
- Confusion Matrix Interpretation
- Train/Test Split with Stratification
- Data Leakage Prevention
- Model Persistence (Serialization)
- Web Application Development

---

*AIML Academic Project — Customer Support Query Classification*
