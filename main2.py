import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# 1. Load the dataset
# Replace the file path with the location where you've saved the dataset
file_path = 'spambase.data'  # Modify this path accordingly
columns = [f'feature{i}' for i in range(1, 58)] + ['label']  # Columns for features and target variable
df = pd.read_csv(file_path, header=None, names=columns)

# 2. Data Preprocessing
# Handling missing values (if any)
print("Missing values:\n", df.isnull().sum())

# Check for duplicates
print("\nDuplicates in the dataset:", df.duplicated().sum())

# Check the first few records
print("\nData Preview:")
print(df.head())

# Feature and target separation
X = df.drop('label', axis=1)  # Features
y = df['label']  # Target (label)

# 3. Data Visualization
# Visualizing feature correlations
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm', fmt='.1f', linewidths=0.5)
plt.title("Feature Correlations")
plt.show()

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Decision Tree Classifier (Rule-based AI)
# Initialize the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# 6. Model Prediction
y_pred = clf.predict(X_test)

# 7. Model Evaluation
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Handling Overfitting & Underfitting
# Cross-validation to check for overfitting/underfitting
cross_val_scores = cross_val_score(clf, X, y, cv=10)
print(f'\nCross-validation scores: {cross_val_scores}')
print(f'Average Cross-validation score: {cross_val_scores.mean()}')

# 9. Feature Importance (Optional, to see which features are most important)
feature_importances = clf.feature_importances_
indices = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.barh(range(X.shape[1]), feature_importances[indices], align='center')
plt.yticks(range(X.shape[1]), [f'feature{i+1}' for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# 10. Handling Overfitting with Pruning (Optional)
# You can prune the decision tree using parameters like max_depth to avoid overfitting
clf_pruned = DecisionTreeClassifier(random_state=42, max_depth=5)
clf_pruned.fit(X_train, y_train)
y_pred_pruned = clf_pruned.predict(X_test)

# Pruned Model Evaluation
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print(f'\nAccuracy of model: {accuracy_pruned * 100:.2f}%')
