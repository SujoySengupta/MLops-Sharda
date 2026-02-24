import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading data...")
raw_data = load_breast_cancer()
df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
df['target'] = raw_data.target

X = df.drop('target', axis=1)
y = df['target']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

print("\nEvaluating model...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("-" * 30)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)
print("Classification Report:")
print(classification_report(y_test, predictions))

model_filename = 'logistic_regression_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved successfully as '{model_filename}'")