import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (works without display)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from nlp_utils import preprocess_text

# Optional speed optimization: set to None to use full dataset
# If training is slow on your laptop, change to e.g. 20000
MAX_ROWS = None

# File paths
CSV_PATH = "data/Combined Data.csv"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MODEL_PATH = "models/logistic_model.pkl"


def main():
    print("=" * 60)
    print("Mental Health Detection - Training Pipeline")
    print("=" * 60)

    # 1. Check if dataset exists
    if not os.path.exists(CSV_PATH):
        print(f"\n[ERROR] Dataset not found at: {CSV_PATH}")
        print("Please download Combined Data.csv from Kaggle and place it inside the data folder.")
        print("Kaggle dataset: suchintikasarkar/sentiment-analysis-for-mental-health")
        return

    # 2. Load dataset
    print(f"\nLoading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # 3. Handle column names
    # The Kaggle dataset has columns: statement and status (+ maybe an ID column)
    # We need to keep only statement and status

    # Normalize column names (strip spaces, lower)
    df.columns = [c.strip() for c in df.columns]

    # Check for expected columns
    # Handle case sensitivity and variations
    col_lower = {c.lower(): c for c in df.columns}

    # Find statement column
    if "statement" in col_lower:
        statement_col = col_lower["statement"]
    elif "text" in col_lower:
        statement_col = col_lower["text"]
    else:
        print(f"[ERROR] Could not find 'statement' column. Found: {list(df.columns)}")
        return

    # Find status column
    if "status" in col_lower:
        status_col = col_lower["status"]
    elif "label" in col_lower:
        status_col = col_lower["label"]
    else:
        print(f"[ERROR] Could not find 'status' column. Found: {list(df.columns)}")
        return

    # Keep only required columns and rename to text/label
    df = df[[statement_col, status_col]]
    df.columns = ["text", "label"]

    print(f"\nUsing columns: text (from {statement_col}), label (from {status_col})")
    print(f"Shape after selecting required columns: {df.shape}")

    # 4. Clean data
    print("\nCleaning data...")

    # Remove rows where text is empty or label is missing
    initial_rows = len(df)
    df = df.dropna(subset=["text", "label"])

    # Remove rows where text is empty string or only whitespace
    df = df[df["text"].astype(str).str.strip() != ""]

    # Remove duplicate text rows
    df = df.drop_duplicates(subset=["text"])

    print(f"Removed {initial_rows - len(df)} rows (empty/missing/duplicate)")
    print(f"Shape after cleaning: {df.shape}")

    # Show basic dataset info
    print(f"\nNumber of rows: {len(df)}")
    print(f"Number of classes: {df['label'].nunique()}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # Optional sampling for speed
    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        print(f"\nSampling {MAX_ROWS} rows (stratified) for faster training...")
        # Stratified sample
        df, _ = train_test_split(
            df,
            train_size=MAX_ROWS,
            random_state=42,
            stratify=df["label"]
        )
        print(f"Shape after sampling: {df.shape}")
        print(df["label"].value_counts())

    # 5. Preprocess text
    print("\nPreprocessing text (this may take a minute)...")
    # Apply preprocessing function to each text
    df["clean_text"] = df["text"].astype(str).apply(preprocess_text)

    # Remove rows where clean_text became empty after preprocessing
    df = df[df["clean_text"].str.strip() != ""]
    print(f"Shape after preprocessing: {df.shape}")
    print("Sample cleaned text:")
    print(df[["text", "clean_text"]].head(3).to_string())

    # Prepare features and labels
    X = df["clean_text"]
    y = df["label"]

    # 6. Train-test split
    # Split BEFORE fitting TF-IDF to avoid data leakage
    # TF-IDF must be fitted only on training data
    print("\nSplitting data (test_size=0.2, stratify)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 7. TF-IDF Vectorizer
    # Convert text into numerical features
    # TF-IDF gives higher weight to useful words and lower weight to very common words
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    # Fit on training data only, then transform both
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF train shape: {X_train_tfidf.shape}")
    print(f"TF-IDF test shape: {X_test_tfidf.shape}")

    # 8. Train Logistic Regression
    # Logistic Regression is used for multiclass text classification
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=300,
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    print("Training completed!")

    # 9. Evaluation
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    # Classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

    # Save metrics to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.txt", "w", encoding="utf-8") as f:
        f.write("Mental Health Detection - Model Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")
        # Also add precision, recall per idea
        f.write(f"\nDataset rows used: {len(df)}\n")
        f.write(f"Classes: {list(model.classes_)}\n")

    print("\nSaved metrics to outputs/metrics.txt")

    # Confusion Matrix
    print("Creating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved outputs/confusion_matrix.png")

    # Class distribution plot
    print("Creating class distribution plot...")
    counts = df["label"].value_counts()
    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Class Distribution")
    plt.xlabel("Mental Health Status")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha="right")
    # Add count labels on bars
    for i, v in enumerate(counts):
        plt.text(i, v + 50, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/class_distribution.png", dpi=150)
    plt.close()
    print("Saved outputs/class_distribution.png")

    # 10. Save model and vectorizer
    # Save the trained model so that Streamlit does not train it again
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model to {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("Now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
