import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
np.random.seed(42)
num_samples = 1000
symptoms = ['fever', 'cough', 'fatigue', 'headache', 'nausea', 'diarrhea', 'sore_throat', 'shortness_of_breath']
data = np.random.randint(0, 2, size=(num_samples, len(symptoms)))
df = pd.DataFrame(data, columns=symptoms)
diagnoses = ['Flu', 'Common Cold', 'COVID-19', 'Gastroenteritis', 'Migraine']
df['diagnosis'] = np.random.choice(diagnoses, size=num_samples)
df.to_csv('health_data.csv', index=False)
data = pd.read_csv('health_data.csv')
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])
X = data[symptoms]
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
new_patient = pd.DataFrame({
    'fever': [1],
    'cough': [0],
    'fatigue': [0],
    'headache': [1],
    'nausea': [0],
    'diarrhea': [1],
    'sore_throat': [1],
    'shortness_of_breath': [0]
})

predicted_diagnosis = model.predict(new_patient)
print(f"Predicted Diagnosis: {le.inverse_transform(predicted_diagnosis)[0]}")