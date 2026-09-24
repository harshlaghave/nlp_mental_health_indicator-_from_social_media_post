# Mental Health Detection from Social Media Posts

## Project Description

This is an NLP-based academic screening demonstration that classifies social-media-style text into mental-health-related categories using classical machine learning.

It is NOT a medical diagnosis system. It is a college mini-project to demonstrate the NLP pipeline: text preprocessing, TF-IDF feature extraction, and classification with Logistic Regression.

We initially considered deep learning models such as BERT and RoBERTa, but for this mini project we selected TF-IDF with Logistic Regression because it is faster, lightweight, easier to train on a CPU, and easier to interpret for an academic NLP demonstration.

---

## Dataset

**Kaggle:** Sentiment Analysis for Mental Health  
**Kaggle identifier:** `suchintikasarkar/sentiment-analysis-for-mental-health`

Original important columns:
- `statement` — input text
- `status` — target label

Categories (7 classes):
- Normal
- Depression
- Suicidal
- Anxiety
- Stress
- Bi-Polar
- Personality Disorder

> The dataset is compiled from multiple public sources and is used here only for educational NLP experimentation. The project automatically handles an extra ID/index column if present and uses only `statement` and `status`.

Place the CSV here after downloading: `data/Combined Data.csv`

---

## Technologies

- Python
- Pandas
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Matplotlib
- Joblib

---

## NLP Pipeline

```
Raw Text
  ↓
Cleaning (lowercase, remove URLs, mentions, punctuation)
  ↓
Tokenization
  ↓
Stop-word Removal
  ↓
Lemmatization
  ↓
TF-IDF Feature Extraction
  ↓
Logistic Regression Classification
  ↓
Prediction + Probability + Feature Contribution
```

---

## Installation (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset Setup

1. Download the Kaggle dataset: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
2. Extract the CSV file
3. Rename it to `Combined Data.csv` if needed
4. Place it inside the `data` folder: `data/Combined Data.csv`

The project expects exactly: `data/Combined Data.csv`

---

## Train

```bat
python train_model.py
```

This will:
- Load and clean the dataset
- Preprocess text
- Split data (stratified, test_size=0.2)
- Fit TF-IDF on training data only
- Train Logistic Regression
- Evaluate (accuracy, classification report, confusion matrix)
- Save models to `models/` and outputs to `outputs/`

> The model must be trained BEFORE launching the application.

---

## Run Web App

```bat
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

---

## What Each File Does

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web app — loads saved model/vectorizer, takes user text, shows prediction, confidence, all probabilities, important words, highlighted text |
| `train_model.py` | Training pipeline — loads CSV, cleans data, preprocesses, splits, TF-IDF, trains Logistic Regression, evaluates and saves models/outputs |
| `nlp_utils.py` | Preprocessing function — lowercase, URL/mention removal, punctuation cleaning, tokenization, stopword removal, lemmatization |

---

## Project Structure

```
mental_health_detection/
│
├── app.py
├── train_model.py
├── nlp_utils.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Combined Data.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── logistic_model.pkl
│
└── outputs/
    ├── confusion_matrix.png
    ├── class_distribution.png
    └── metrics.txt
```

---

## Model Details

- **TF-IDF Vectorizer:**
  ```python
  TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.95)
  ```
- **Classifier:**
  ```python
  LogisticRegression(max_iter=300, random_state=42)
  ```
- **Split:** `train_test_split(test_size=0.2, random_state=42, stratify=y)`

TF-IDF is fitted only on training data to prevent data leakage.

---

## Important Limitations

- Social-media language is noisy and informal.
- Labels come from the dataset and may contain annotation limitations.
- Mental-health conditions cannot be reliably diagnosed from text alone.
- Model can produce incorrect predictions.
- Class imbalance can affect results (some categories have many more samples than others).
- The model learns patterns from training data rather than understanding a person's actual mental state.
- The dataset combines multiple source datasets, so writing style varies across samples.

---

## Disclaimer

This application is an academic NLP demonstration. It does not diagnose mental health conditions, and model predictions should not be treated as medical advice. For any mental health concerns, please consult a qualified professional.

---

## Viva Questions

**1. What is NLP?**  
NLP (Natural Language Processing) is a field of AI that helps computers understand, process, and analyze human language. It deals with text and speech data. Example tasks are classification, translation, and sentiment analysis.

**2. What is text preprocessing?**  
Text preprocessing is cleaning and preparing raw text before giving it to a model. It includes lowercasing, removing noise like URLs, tokenizing, removing stopwords, and lemmatizing. It helps the model focus on meaningful words.

**3. Why do we convert text to lowercase?**  
To treat same words with different cases as identical, e.g., "Stress" and "stress" become the same. This reduces vocabulary size and avoids duplicate features. It makes the model more consistent.

**4. What is tokenization?**  
Tokenization is splitting a sentence into individual words or tokens. For example, "I feel anxious" becomes ["I", "feel", "anxious"]. It is the first step to convert text into a form the model can process.

**5. What are stop words?**  
Stop words are very common words like "is", "the", "and", "a" that occur frequently but carry little meaning. Removing them reduces noise and helps the model focus on important words. NLTK provides a list of English stopwords.

**6. What is lemmatization?**  
Lemmatization converts a word to its base or dictionary form. For example, "running" becomes "run" and "better" becomes "good". It groups different forms of the same word together.

**7. What is TF-IDF?**  
TF-IDF (Term Frequency - Inverse Document Frequency) converts text into numbers. TF measures how often a word appears in a document, IDF reduces weight of very common words and increases weight of rare, informative words. Together they give important words higher scores.

**8. Why is TF-IDF better than simple word counting for this project?**  
Simple counting (Bag of Words) gives equal importance to all words, even common ones. TF-IDF down-weights very common words that appear in many documents and highlights discriminative words. This improves classification for text data.

**9. What is Logistic Regression?**  
Logistic Regression is a supervised ML algorithm used for classification. Despite its name, it predicts probabilities for each class using a logistic (sigmoid) function and picks the class with highest probability. It works well with high-dimensional sparse data like TF-IDF.

**10. Why did you choose Logistic Regression?**  
It is simple, fast, works well for multiclass text classification, supports predict_proba(), and is easy to explain. Unlike BERT or deep models, it trains quickly on CPU without GPU. It is also interpretable because we can check coefficients to find important words.

**11. What is train-test split?**  
Train-test split divides the dataset into two parts: one for training the model and one for testing its performance on unseen data. In this project we use 80% for training and 20% for testing. This checks whether the model generalizes well.

**12. Why use stratify?**  
Stratify keeps the same class proportion in both train and test sets as in the original dataset. This is important when classes are imbalanced, so every class appears in both sets fairly. Without it, a rare class might be missing from the test set.

**13. What does random_state=42 mean?**  
random_state fixes the random seed so that shuffling and splitting give the same result every time the code runs. This makes experiments reproducible. The number 42 is just a commonly used arbitrary choice.

**14. What does predict_proba() do?**  
predict_proba() returns the model's predicted probability for each class instead of just the final predicted class. For example, it might say 70% Depression, 10% Anxiety, etc. We use it to show confidence and all class probabilities in the app.

**15. How are important words identified?**  
We use TF-IDF value × Logistic Regression coefficient for the predicted class. Each word has a TF-IDF score and the model has a coefficient (weight) for that word. Multiplying them gives contribution; words with highest positive contribution are shown as important and highlighted in the input.

---

## Exact Commands to Run (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

---

## License

Academic project — free for educational use.
