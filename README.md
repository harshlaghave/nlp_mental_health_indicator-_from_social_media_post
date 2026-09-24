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
