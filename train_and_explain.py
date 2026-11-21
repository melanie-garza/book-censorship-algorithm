import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report

# 1. LOAD & CLEAN DATA
CSV_PATH = "merged_dataset.csv"
df = pd.read_csv(CSV_PATH)

# Normalize column names to lowercase just in case
df.columns = [c.strip().lower() for c in df.columns]
print("Columns detected:", df.columns.tolist())

COL_TITLE = "title"
COL_DESC = "description"
COL_GENRE = "genre"
COL_TARGET = "banned"

# Drop rows with missing text or target
df = df.dropna(subset=[COL_DESC, COL_GENRE, COL_TARGET]).copy()
df[COL_TARGET] = df[COL_TARGET].astype(int)

print(f"Dataset size after cleaning: {len(df)} books")
print(df[COL_TARGET].value_counts(normalize=True).round(3))

# Train/Test Split (80% train, 20% test, stratified to keep class balance)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[COL_TARGET]
)

# ==============================================================================
# MODEL 1: LOGISTIC REGRESSION (on Book Descriptions via TF-IDF)
# ==============================================================================
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=3000,   # Top 3,000 most informative words
    ngram_range=(1, 2)   # Captures single words and 2-word phrases
)

X_train_text = tfidf.fit_transform(train_df[COL_DESC])
X_test_text = tfidf.transform(test_df[COL_DESC])
y_train = train_df[COL_TARGET]
y_test = test_df[COL_TARGET]

# C=1.0 controls regularization (prevents memorizing rare words)
log_reg = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_text, y_train)

print("\n--- LOGISTIC REGRESSION REPORT ---")
print(classification_report(y_test, log_reg.predict(X_test_text)))

# Plot Top 15 Words pushing toward Banned (+) vs Not Banned (-)
feature_names = np.array(tfidf.get_feature_names_out())
coefs = log_reg.coef_[0]

top_pos_idx = np.argsort(coefs)[-15:]
top_neg_idx = np.argsort(coefs)[:15]
top_idx = np.concatenate([top_neg_idx, top_pos_idx])

plt.figure(figsize=(10, 7))
colors = ["#2b7bba" if c < 0 else "#d62728" for c in coefs[top_idx]]
plt.barh(feature_names[top_idx], coefs[top_idx], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Logistic Regression: Top Word Weights Driving Book Censorship")
plt.xlabel("Coefficient Weight (Negative = Not Banned | Positive = Banned)")
plt.tight_layout()
plt.savefig("logistic_weights.png", dpi=150)
plt.close()

# ==============================================================================
# FUNCTION TO EXPLAIN A SINGLE BOOK (FOR YOUR VIDEO DEMO)
# ==============================================================================
def explain_book(book_title_query):
    matches = df[df[COL_TITLE].str.contains(book_title_query, case=False, na=False)]
    if matches.empty:
        print(f"No book found matching '{book_title_query}'")
        return
    
    book = matches.iloc[0]
    vec = tfidf.transform([book[COL_DESC]])
    prob = log_reg.predict_proba(vec)[0][1]
    
    # Calculate individual word contributions: weight * tfidf_value
    contributions = vec.toarray()[0] * coefs
    non_zero = np.where(contributions != 0)[0]
    
    # Sort by absolute impact
    sorted_idx = non_zero[np.argsort(np.abs(contributions[non_zero]))[::-1][:8]]
    
    print(f"\n=== INTERNAL MATH FOR: {book[COL_TITLE]} ===")
    print(f"Actual Label: {'BANNED' if book[COL_TARGET] == 1 else 'NOT BANNED'}")
    print(f"Model Predicted Probability of Ban: {prob*100:.1f}%")
    print(f"Intercept (bias b): {log_reg.intercept_[0]:.3f}")
    print("Top contributing words in its description:")
    for idx in sorted_idx:
        sign = "+" if contributions[idx] > 0 else ""
        print(f"  '{feature_names[idx]}': {sign}{contributions[idx]:.4f}")

# Test it on a classic!
explain_book("Fahrenheit 451")

# ==============================================================================
# MODEL 2: DECISION TREE (on Genres for Clear If/Else Rules)
# ==============================================================================
# Convert genre strings into binary presence/absence features
genre_vec = CountVectorizer(binary=True, max_features=40, stop_words="english")
X_train_genre = genre_vec.fit_transform(train_df[COL_GENRE])
X_test_genre = genre_vec.transform(test_df[COL_GENRE])

# max_depth=3 keeps the tree interpretable and prevents overfitting
tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
tree.fit(X_train_genre, y_train)

print("\n--- DECISION TREE REPORT (GENRES ONLY) ---")
print(classification_report(y_test, tree.predict(X_test_genre)))

plt.figure(figsize=(18, 9))
plot_tree(
    tree,
    feature_names=genre_vec.get_feature_names_out(),
    class_names=["Not Banned", "Banned"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree (max_depth=3): Genre Rules Behind Book Bans")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.close()

print("\nDone! Check 'logistic_weights.png' and 'decision_tree.png'.")