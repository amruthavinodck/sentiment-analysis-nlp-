"""
Sentiment Analysis on Customer Feedback (NLP)
================================================
Amrutha Vinod

Goal: Automatically classify customer reviews as Positive, Negative, or
Neutral, so product/support teams get fast, scalable visibility into
customer sentiment instead of manually reading thousands of reviews.

Pipeline: Load -> Clean text -> TF-IDF vectorize -> Train/Test split ->
          Multinomial Naive Bayes + Logistic Regression -> Evaluate ->
          Inspect most sentiment-driving words
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sns.set_style("whitegrid")
OUT = "/home/claude/sentiment_project"

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv(f"{OUT}/customer_feedback.csv")
print("Shape:", df.shape)
print(df["sentiment"].value_counts())

# ---------------------------------------------------------------
# 2. TEXT CLEANING
# ---------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text

df["clean_text"] = df["review_text"].apply(clean_text)
print("\nExample cleaning:")
print("Before:", df["review_text"].iloc[0])
print("After :", df["clean_text"].iloc[0])

# ---------------------------------------------------------------
# 3. EDA — class balance and review length
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).plot(
    kind="bar", ax=axes[0], color=["#2E86AB", "#F4A261", "#E63946"]
)
axes[0].set_title("Sentiment Class Distribution")
axes[0].tick_params(axis="x", rotation=0)

df["review_length"] = df["clean_text"].apply(lambda x: len(x.split()))
sns.boxplot(data=df, x="sentiment", y="review_length", ax=axes[1],
            order=["Positive", "Neutral", "Negative"],
            palette=["#2E86AB", "#F4A261", "#E63946"])
axes[1].set_title("Review Length by Sentiment")

plt.tight_layout()
plt.savefig(f"{OUT}/eda_overview.png", dpi=120)
plt.close()
print("\nSaved eda_overview.png")

# ---------------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# ---------------------------------------------------------------
X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------------------------------------------
# 5. TF-IDF VECTORIZATION
# ---------------------------------------------------------------
# TF-IDF weighs words by how distinctive they are to a document, not just
# how often they appear -- common words across all reviews get downweighted.
vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF feature count: {X_train_tfidf.shape[1]}")

# ---------------------------------------------------------------
# 6. MODEL 1 — Multinomial Naive Bayes (classic, fast text baseline)
# ---------------------------------------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)

# ---------------------------------------------------------------
# 7. MODEL 2 — Logistic Regression (usually stronger on TF-IDF features)
# ---------------------------------------------------------------
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_tfidf, y_train)
log_preds = log_model.predict(X_test_tfidf)

# ---------------------------------------------------------------
# 8. EVALUATION
# ---------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 3))
    print("Precision (macro):", round(precision_score(y_true, y_pred, average="macro"), 3))
    print("Recall (macro)   :", round(recall_score(y_true, y_pred, average="macro"), 3))
    print("F1 (macro)       :", round(f1_score(y_true, y_pred, average="macro"), 3))
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

results = [
    evaluate("Multinomial Naive Bayes", y_test, nb_preds),
    evaluate("Logistic Regression", y_test, log_preds),
]
pd.DataFrame(results).to_csv(f"{OUT}/model_results.csv", index=False)

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
labels = ["Positive", "Neutral", "Negative"]
for ax, (name, preds) in zip(axes, [("Naive Bayes", nb_preds), ("Logistic Regression", log_preds)]):
    cm = confusion_matrix(y_test, preds, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(f"{name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrices.png", dpi=120)
plt.close()
print("\nSaved confusion_matrices.png")

# ---------------------------------------------------------------
# 9. MOST SENTIMENT-DRIVING WORDS (Logistic Regression coefficients)
# ---------------------------------------------------------------
feature_names = np.array(vectorizer.get_feature_names_out())

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for idx, cls in enumerate(log_model.classes_):
    coefs = log_model.coef_[idx]
    top_idx = np.argsort(coefs)[-10:]
    axes[idx].barh(feature_names[top_idx], coefs[top_idx], color="#2E86AB")
    axes[idx].set_title(f"Top words driving '{cls}'")
plt.tight_layout()
plt.savefig(f"{OUT}/top_words_per_class.png", dpi=120)
plt.close()
print("Saved top_words_per_class.png")

print("\n\n=== DONE: outputs saved in", OUT, "===")
