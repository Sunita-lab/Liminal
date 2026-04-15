import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/headlines_week1.csv")
pd.set_option("display.max_columns", None)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nUnique domain values before cleaning:")
print("\nDomain:", df["domain"].unique())

#Clean domain column
df["domain"] = df["domain"].str.strip().str.lower()

print("\nDomain Counts after cleaning:")
print(df["domain"].value_counts())
print("\nDate Range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("\nUnique Sources:", df["source"].nunique())

# Feature Engineering

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Basic text features
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
df["char_count"] = df["text"].apply(lambda x: len(str(x)))
df["has_question"] = df["text"].str.contains("\?")
df["has_number"] = df["text"].str.contains(r"\d")

print("\nAverage word count by domain:")
print(df.groupby("domain")["word_count"].mean())

print("\nAverage character count by domain:")
print(df.groupby("domain")["char_count"].mean())

print("\nQuestion headline percentage by domain:")
print(df.groupby("domain")["has_question"].mean() * 100)

print("\nNumeric headline percentage by domain:")
print(df.groupby("domain")["has_number"].mean() * 100)

print("\nArticles per day by domain:")
daily_counts = df.groupby(["timestamp", "domain"]).size().unstack()
print(daily_counts.fillna(0))

# Volatility measure (standard deviation of daily counts)
volatility = daily_counts.std()

print("\nVolatility (Std Dev) by domain:")
print(volatility)

# Identify spike days (above mean + 1 std)
thresholds = daily_counts.mean() + daily_counts.std()

print("\nSpike Thresholds:")
print(thresholds)

spikes = daily_counts[daily_counts > thresholds]

print("\nSpike Days:")
print(spikes.dropna(how="all"))

# Burstiness = std / mean
burstiness = daily_counts.std() / daily_counts.mean()

print("\nBurstiness Index:")
print(burstiness)

import matplotlib.pyplot as plt

daily_counts.plot(figsize=(10,5))
plt.title("Daily Article Count by Domain")
plt.xlabel("Date")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("daily_counts.png")
plt.close()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Select features
X = df[["word_count", "char_count", "has_question", "has_number"]]
y = df["domain"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from textblob import TextBlob

# Simple sentiment score
df["sentiment"] = df["text"].apply(lambda x: TextBlob(x).sentiment.polarity)

print("\nAverage Sentiment by Domain:")
print(df.groupby("domain")["sentiment"].mean())

print("\nSentiment Std Dev by Domain:")
print(df.groupby("domain")["sentiment"].std())

from collections import Counter
import re

def get_top_words(text_series, n=15):
    words = []
    for text in text_series:
        tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        words.extend(tokens)
    return Counter(words).most_common(n)

print("\nTop Words - Environment:")
print(get_top_words(df[df["domain"]=="environment"]["text"]))

print("\nTop Words - Entertainment:")
print(get_top_words(df[df["domain"]=="entertainment"]["text"]))

def vocabulary_size(text_series):
    words = set()
    for text in text_series:
        tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        words.update(tokens)
    return len(words)

print("\nVocabulary Size - Environment:")
print(vocabulary_size(df[df["domain"]=="environment"]["text"]))

print("\nVocabulary Size - Entertainment:")
print(vocabulary_size(df[df["domain"]=="entertainment"]["text"]))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

X = df["text"]
y = df["domain"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\nTF-IDF Classification Report:")
print(classification_report(y_test, y_pred))

# Create daily summary
daily_summary = df.groupby(["timestamp", "domain"]).size().unstack().fillna(0)

daily_summary["ent_rolling_mean"] = daily_summary["entertainment"].rolling(3).mean()
daily_summary["env_rolling_mean"] = daily_summary["environment"].rolling(3).mean()

daily_summary["ent_rolling_std"] = daily_summary["entertainment"].rolling(3).std()
daily_summary["env_rolling_std"] = daily_summary["environment"].rolling(3).std()

print(daily_summary)

# Define spike thresholds again
ent_threshold = daily_summary["entertainment"].mean() + 1.5*daily_summary["entertainment"].std()
env_threshold = daily_summary["environment"].mean() + daily_summary["environment"].std()

# Create spike labels
daily_summary["ent_spike"] = (daily_summary["entertainment"] > ent_threshold).astype(int)
daily_summary["env_spike"] = (daily_summary["environment"] > env_threshold).astype(int)

print("\nSpike Labels:")
print(daily_summary[["entertainment", "ent_spike", "environment", "env_spike"]])

# Create lag features (previous day’s rolling stats)
daily_summary["ent_mean_lag1"] = daily_summary["ent_rolling_mean"].shift(1)
daily_summary["ent_std_lag1"] = daily_summary["ent_rolling_std"].shift(1)

# Drop rows with NaN
daily_model_data = daily_summary.dropna()

print("\nModel Data Preview:")
print(daily_model_data[[
    "entertainment",
    "ent_spike",
    "ent_mean_lag1",
    "ent_std_lag1"
]])

from sklearn.linear_model import LogisticRegression

X = daily_model_data[["ent_mean_lag1", "ent_std_lag1"]]
y = daily_model_data["ent_spike"]

model = LogisticRegression()
model.fit(X, y)

preds = model.predict(X)

print("\nPredictions:", preds)
print("Actual:", y.values)

# --- Physical Threat Index (PTI) ---

threat_words = [
    "death", "deaths", "killed", "kill", "disaster", "crisis",
    "collapse", "flood", "wildfire", "heatwave", "toxic",
    "pollution", "disease", "risk", "warning", "emergency",
    "warming", "extinction", "carbon", "temperature",
    "toxic", "contamination", "hazard", "mortality",
    "drought", "storm", "cyclone", "heat"
]

import re

def compute_pti(text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(tokens) == 0:
        return 0
    
    threat_count = sum(1 for word in tokens if word in threat_words)
    
    return threat_count / len(tokens) if len(tokens) > 0 else 0

df["pti"] = df["text"].apply(compute_pti)

print("\nAverage PTI by Domain:")
print(df.groupby("domain")["pti"].mean())

print("\nPTI Std Dev by Domain:")
print(df.groupby("domain")["pti"].std())

daily_pti = df.groupby(["timestamp", "domain"])["pti"].mean().unstack().fillna(0)

print("\nDaily Average PTI:")
print(daily_pti)