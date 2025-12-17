import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')

#Load data
car_df = pd.read_csv(r'C:\Users\valen\OneDrive\Desktop\Car_Reviews_Database.csv', encoding='latin1')
bike_df = pd.read_csv(r'C:\Users\valen\OneDrive\Desktop\bike_rental_reviews.csv', encoding='latin1')

#Combine datasets
df = pd.concat([car_df, bike_df], ignore_index=True)

print("Data loaded. Shape:", df.shape)
print("Columns:", df.columns)

#Detect review column
if 'review_text' in df.columns:
    review_col = 'review_text'
elif 'Review' in df.columns:
    review_col = 'Review'
else:
    raise ValueError("No review text column found!")

#Drop rows with missing or empty sentiment or review
df = df.dropna(subset=[review_col, 'sentiment']).reset_index(drop=True)
df = df[df['sentiment'].astype(str).str.strip() != '']
df = df[df[review_col].astype(str).str.strip() != '']

#Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

df['clean_review'] = df[review_col].apply(clean_text)

#Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("\nNaive Bayes Results:")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))

#Wordcloud for topic modeling
for sentiment in ['positive', 'neutral', 'negative']:
    text = ' '.join(df[df['sentiment'] == sentiment]['clean_review'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for {sentiment.capitalize()} Reviews")
    plt.show()

#Topic modeling with LDA
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda_topics = lda.fit_transform(X)

def print_top_words(model, feature_names, n_top_words=10):
    for idx, topic in enumerate(model.components_):
        print(f"\nTopic #{idx+1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

print("\nLDA Topic Modeling:")
print_top_words(lda, vectorizer.get_feature_names_out())

#Sample prediction visualization
sample_idx = np.random.choice(X_test.shape[0], 5, replace=False)
sample_reviews = df.iloc[y_test.index[sample_idx]][review_col]
sample_preds = lr.predict(X_test[sample_idx])

for review, pred in zip(sample_reviews, sample_preds):
    print(f"\nReview: {review}\nPredicted Sentiment: {pred}")

print("\nPipeline complete. You can further extend this with LSTM/BERT using TensorFlow or HuggingFace Transformers if desired.")