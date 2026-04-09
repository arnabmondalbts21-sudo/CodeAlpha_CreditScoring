import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample medical data তৈরি
X, y = make_classification(n_samples=1000, n_features=8, random_state=42)

columns = ['age', 'blood_pressure', 'cholesterol', 'heart_rate',
           'glucose', 'bmi', 'smoking', 'exercise']

df = pd.DataFrame(X, columns=columns)
df['disease'] = y

# Split
X = df.drop('disease', axis=1)
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Report
print("===== REPORT =====")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Reds')
plt.title("Disease Prediction Model")
plt.show()