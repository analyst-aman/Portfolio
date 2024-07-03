import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

# Identify top 10 indications
top_10_indications = data['Indication'].value_counts().nlargest(10).index

# Filter the dataset for top 10 indications
data_top_10 = data[data['Indication'].isin(top_10_indications)]

# Select features and target variable
X = data_top_10[['Age', 'Gender', 'Name of Drug', 'Route', 'Indication', 'Duration (days)', 'Frequency']]
y = data_top_10['Dosage (gram)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on the test data
y_pred = lr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plotting the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Dosage (gram)')
plt.ylabel('Predicted Dosage (gram)')
plt.title('Actual vs Predicted Dosage')
plt.show()
