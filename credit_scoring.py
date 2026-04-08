import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# data making
X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, random_state=42)

feature_names = ['income', 'debt', 'payment_history', 'age', 'employment_years',
                 'num_loans', 'credit_limit', 'monthly_expense', 'savings', 'num_defaults']

df = pd.DataFrame(X, columns=feature_names)
df['creditworthy'] = y

# separate
X = df.drop('creditworthy', axis=1)
y = df['creditworthy']

# Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Result
print("===== REPORT =====")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title("Credit Scoring Model")
plt.show()