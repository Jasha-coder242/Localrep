import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample dataset
df = pd.DataFrame({
    'Text': [
        'Cats are running quickly.',
        'Dogs were barking loudly .',
        'He studies in university.',
        'They are enjoying the holidays'
    ]
})

print("Original DataFrame:")
print(df)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Fit and transform the Text column
X = tfidf.fit_transform(df["Text"])

# Convert TF-IDF output into a DataFrame
tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

print("\nTF-IDF Representation:")
print(tfidf_df)