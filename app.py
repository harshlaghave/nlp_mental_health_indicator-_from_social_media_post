import os
import re
import html
import joblib
import streamlit as st
import pandas as pd

from nlp_utils import preprocess_text

# Page config
st.set_page_config(
    page_title="Mental Health Detection",
    page_icon="🧠",
    layout="wide"
)

# File paths for saved models
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MODEL_PATH = "models/logistic_model.pkl"

# Load model and vectorizer with caching
@st.cache_resource
def load_models():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def get_top_features(vectorizer, model, input_tfidf, predicted_class, top_n=5):
    """
    Find important words for predicted class
    Contribution = TF-IDF value * Logistic Regression coefficient
    """
    try:
        # Get feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()

        # Find index of predicted class
        class_index = list(model.classes_).index(predicted_class)

        # Get coefficients for predicted class
        # For multiclass, coef_ shape is (n_classes, n_features)
        coefficients = model.coef_[class_index]

        # Get TF-IDF values for input (sparse matrix to dense array)
        tfidf_array = input_tfidf.toarray()[0]

        # Calculate contribution: tfidf * coefficient
        contributions = tfidf_array * coefficients

        # Get indices of features that actually appeared in input (tfidf > 0)
        # And sort by contribution (highest positive first)
        feature_contributions = []
        for i, tfidf_val in enumerate(tfidf_array):
            if tfidf_val > 0:
                feature_contributions.append(
                    (feature_names[i], contributions[i], tfidf_val, coefficients[i])
                )

        # Sort by contribution descending
        feature_contributions.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return feature_contributions[:top_n]

    except Exception as e:
        return []


def highlight_text(original_text, important_words):
    """
    Highlight important words in original text using <mark> tags
    Escape HTML first to prevent injection
    """
    # Escape user input to prevent HTML injection
    escaped = html.escape(original_text)

    # For each important word, highlight it (case-insensitive)
    for word, _, _, _ in important_words:
        # Word may be a phrase (bigram), handle both
        # Use regex with word boundaries for single words
        if " " in word:
            # Phrase - simple case-insensitive replacement
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            escaped = pattern.sub(f"<mark>{word}</mark>", escaped)
        else:
            pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
            escaped = pattern.sub(f"<mark>{word}</mark>", escaped)

    # Also add style for mark tag
    highlighted = f'<p style="line-height:1.8; font-size:16px;">{escaped}</p>'
    return highlighted


# Sidebar
with st.sidebar:
    st.title("About Project")
    st.markdown("""
**NLP pipeline:**  
Text → Preprocessing → TF-IDF → Logistic Regression → Prediction

**Dataset:**  
Kaggle — Sentiment Analysis for Mental Health

**Model:**  
TF-IDF + Logistic Regression

**Classes:**  
7 (Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, Personality Disorder)
    """)

    st.divider()
    st.subheader("Disclaimer")
    st.warning(
        "This application is an academic NLP demonstration. "
        "It does not diagnose mental health conditions, and model predictions "
        "should not be treated as medical advice. "
        "Predictions may be incorrect and should not be used for clinical decision-making."
    )

    st.divider()
    st.subheader("How the system works")
    st.markdown("""
1. **Text Cleaning** — lowercasing, removing URLs, mentions, punctuation
2. **Tokenization** — splitting text into words
3. **Stop-word Removal** — removing common words like 'the', 'is'
4. **Lemmatization** — converting words to base form (e.g., running → run)
5. **TF-IDF Feature Extraction** — converting words to numerical importance scores
6. **Logistic Regression Classification** — predicting the class
7. **Probability Calculation** — predict_proba gives confidence for each class
8. **Feature Contribution Analysis** — TF-IDF × coefficient to find important words
    """)

# Main content
st.title("Mental Health Detection from Social Media Text")
st.markdown("*NLP-based academic screening demonstration using TF-IDF and Logistic Regression*")
st.divider()

# Check if models exist before showing input
if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please run: `python train_model.py`")
    st.info("Make sure you have placed `Combined Data.csv` inside the `data` folder and then run training.")
    st.stop()

# Load models
try:
    vectorizer, model = load_models()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please run `python train_model.py` to train the model again.")
    st.stop()

# Sample buttons (simple)
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)

# Use session state to handle example clicks simply
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

def set_example(text):
    st.session_state.input_text = text

with col1:
    if st.button("Example: Anxious"):
        set_example("I have been feeling very anxious and worried about everything.")
with col2:
    if st.button("Example: Normal"):
        set_example("I am happy with my life and things are going well.")
with col3:
    if st.button("Example: Stressed"):
        set_example("I feel extremely stressed and exhausted because of work.")

# Input area
st.subheader("Enter a social media post or sentence")
user_input = st.text_area(
    "Enter a social media post or sentence",
    value=st.session_state.input_text,
    height=180,
    placeholder="Type your text here... e.g., I have been under a lot of pressure recently and I am unable to sleep.",
    label_visibility="collapsed"
)

st.caption('Example: "I have been under a lot of pressure recently and I am unable to sleep."')

# Analyze button
if st.button("Analyze Text", type="primary", use_container_width=True):

    # Validation
    if not user_input or not user_input.strip():
        st.error("Please enter some text before analyzing.")
        st.stop()

    if len(user_input.strip().split()) < 3:
        st.warning("Please enter a longer sentence for a more meaningful prediction.")
        st.stop()

    # Preprocess
    cleaned = preprocess_text(user_input)

    if not cleaned or not cleaned.strip():
        st.error("No meaningful words found after preprocessing. Please enter a longer, more descriptive sentence.")
        st.stop()

    # Transform using TF-IDF (vectorizer already fitted on training data)
    input_tfidf = vectorizer.transform([cleaned])

    # Predict
    predicted_class = model.predict(input_tfidf)[0]
    probabilities = model.predict_proba(input_tfidf)[0]  # predict_proba gives probability for each class

    # Confidence of predicted class
    max_prob = max(probabilities)
    confidence = max_prob * 100

    # Display result card
    st.divider()
    st.subheader("Prediction Result")

    # Big card using columns
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Predicted Status", value=predicted_class)
        # Add note for Suicidal class
        if predicted_class.lower() == "suicidal":
            st.caption("Text classified by the model as: Suicidal (academic demonstration only, not a diagnosis)")
    with c2:
        st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

    # --- All class probabilities ---
    st.subheader("Predicted class probabilities from the model")
    # Create sorted list of (class, prob)
    class_probs = list(zip(model.classes_, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)

    # Show as dataframe for bar chart
    prob_df = pd.DataFrame(class_probs, columns=["Status", "Probability"])
    prob_df["Probability (%)"] = prob_df["Probability"] * 100

    # Display table
    # Format nicely
    display_df = prob_df.copy()
    display_df["Probability (%)"] = display_df["Probability (%)"].map(lambda x: f"{x:.2f}%")
    st.dataframe(display_df[["Status", "Probability (%)"]], use_container_width=True, hide_index=True)

    # Bar chart
    # Use streamlit bar chart - set Status as index
    chart_df = prob_df.set_index("Status")
    st.bar_chart(chart_df["Probability (%)"])

    # --- Important words ---
    st.subheader("Words contributing to this prediction")
    st.caption("Model-important words/features — TF-IDF value × Logistic Regression coefficient")

    top_features = get_top_features(vectorizer, model, input_tfidf, predicted_class, top_n=5)

    if not top_features or all(c <= 0 for _, c, _, _ in top_features):
        st.info("No strong individual word contribution was found for this short input.")
    else:
        # Filter to only positive contributions
        positive_features = [f for f in top_features if f[1] > 0]
        if not positive_features:
            st.info("No strong individual word contribution was found for this short input.")
        else:
            # Show table
            feat_df = pd.DataFrame(
                [(w, f"{c:+.4f}") for w, c, _, _ in positive_features],
                columns=["Feature", "Contribution"]
            )
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # Also show as bullet list
            st.markdown("**Top contributing terms:**")
            for w, c, _, _ in positive_features:
                st.markdown(f"- `{w}` — contribution: {c:+.4f}")

    # --- Highlighted text ---
    st.subheader("Highlighted Text")
    if top_features:
        highlighted_html = highlight_text(user_input, [f for f in top_features if f[1] > 0])
        # Add custom style for mark
        st.markdown(
            """
            <style>
            mark { background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(highlighted_html, unsafe_allow_html=True)
    else:
        st.markdown(f"<p>{html.escape(user_input)}</p>", unsafe_allow_html=True)

    # Ethical note again
    st.info(
        "This is an academic NLP classification project and not a medical diagnostic system. "
        "Text classified by the model should not be treated as a real diagnosis."
    )

# Footer - always show disclaimer and info
st.divider()
st.markdown("""
<div style="text-align:center; color:gray; font-size:13px;">
This project is for educational purposes only. Mental health conditions cannot be reliably diagnosed from text alone.<br>
If you or someone you know needs help, please contact a mental health professional or helpline in your country.
</div>
""", unsafe_allow_html=True)
