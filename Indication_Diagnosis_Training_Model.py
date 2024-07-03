"""
In this file we will train RandomForestClassifier Model to predict
the diagnosis based on the indications from the hospital dataset.csv

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('Hopsital Dataset.csv')

# Preprocess the data
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data['Dosage (gram)'] = pd.to_numeric(data['Dosage (gram)'], errors='coerce')
data['Duration (days)'] = pd.to_numeric(data['Duration (days)'], errors='coerce')
data['Date of Data Entry'] = pd.to_datetime(data['Date of Data Entry'], dayfirst=True, errors='coerce')
data = data.dropna()

# Label encode categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'Name of Drug', 'Route', 'Indication', 'Frequency']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Encode target variable 'Diagnosis'
le_diagnosis = LabelEncoder()
data['Diagnosis'] = le_diagnosis.fit_transform(data['Diagnosis'])

# Identify top 10 diagnoses
top_10_diagnoses = data['Diagnosis'].value_counts().nlargest(10).index

# Filter the dataset for top 10 diagnoses
data_top_10 = data[data['Diagnosis'].isin(top_10_diagnoses)]

# Select features and target variable
X = data_top_10[['Age', 'Gender', 'Name of Drug', 'Route', 'Indication', 'Duration (days)', 'Frequency']]
y = data_top_10['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=le_diagnosis.inverse_transform(top_10_diagnoses)))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=le_diagnosis.inverse_transform(top_10_diagnoses),
            yticklabels=le_diagnosis.inverse_transform(top_10_diagnoses))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Diagnosis')
plt.ylabel('Actual Diagnosis')
plt.show()
