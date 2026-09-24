import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources if not already present
# This handles cases where student runs code for first time
def download_nltk_resources():
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for res in resources:
        try:
            # punkt and punkt_tab are tokenizers, others are corpora
            if res in ["punkt", "punkt_tab"]:
                nltk.data.find(f"tokenizers/{res}")
            else:
                nltk.data.find(f"corpora/{res}")
        except LookupError:
            try:
                print(f"Downloading NLTK resource: {res} ...")
                nltk.download(res, quiet=True)
            except Exception as e:
                print(f"Could not download {res}: {e}")

# Try to download resources on import
download_nltk_resources()

# Create lemmatizer object
lemmatizer = WordNetLemmatizer()

# Load stopwords once
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean and preprocess input text.
    Steps:
    1. lowercase
    2. remove URLs
    3. remove @mentions
    4. remove punctuation / special characters
    5. tokenize
    6. remove stopwords
    7. lemmatize
    8. join back to string
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove URLs (http, https, www)
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove @mentions (e.g., @username)
    text = re.sub(r"@\w+", "", text)

    # 4. Remove punctuation and special characters
    # Keep only letters and numbers and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # If text becomes empty after cleaning, return empty string
    if not text:
        return ""

    # 5. Tokenize
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        # Fallback if punkt not available
        tokens = text.split()

    # 6. Remove stopwords and 7. Lemmatize
    cleaned_tokens = []
    for word in tokens:
        # Remove stopwords and very short words (1-2 chars are usually not useful)
        if word not in stop_words and len(word) > 1:
            # Lemmatize word (e.g., "running" -> "run")
            lemma = lemmatizer.lemmatize(word)
            cleaned_tokens.append(lemma)

    # 8. Join tokens back into string
    clean_text = " ".join(cleaned_tokens)

    return clean_text
