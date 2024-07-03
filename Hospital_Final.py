"""
This is file for the exploratory data analysis of hospital dataset.csv.
- Used Pandas, pyplot from matplotlib and seaborn.
- Performed data cleaning to get rid of null values.
- Created output CSV files for future reference.



"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
data = pd.read_csv('Hopsital Dataset.csv')

# Setting options to display all columns
pd.set_option('display.max_columns', None)

# Looking at the first few rows of the dataset
print(data.head())

# Preprocessing the data

# Converting Age, Dosage (gram), and Duration (days) to numeric
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data['Dosage (gram)'] = pd.to_numeric(data['Dosage (gram)'], errors='coerce')
data['Duration (days)'] = pd.to_numeric(data['Duration (days)'], errors='coerce')

# Converting Date of Data Entry to datetime
data['Date of Data Entry'] = pd.to_datetime(data['Date of Data Entry'], dayfirst=True,  errors='coerce')

# Checking for null values using .isnull()
null_counts = data.isnull().sum()
print("Null values in each column:\n", null_counts)

# Removing rows with null values using .dropna()
data = data.dropna()

# # Check which columns are numeric
# numeric_columns = data.select_dtypes(include=['number']).columns

# Stats about the data
print(data.describe(include='all'))

# Analyzing different columns using 'pyplot' and 'seaborn'

# Count plot for Route
plt.figure(figsize=(10, 6))
sns.countplot(x='Route', data=data)
plt.title('Distribution of Medication Routes')
plt.xlabel('Route')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Count plot for Frequency
plt.figure(figsize=(10, 6))
sns.countplot(x='Frequency', data=data)
plt.title('Distribution of Medication Frequency')
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Count plot for Diagnosis
plt.figure(figsize=(12, 8))
sns.countplot(y='Diagnosis', data=data, order=data['Diagnosis'].value_counts().index)
plt.title('Distribution of Diagnoses')
plt.xlabel('Count')
plt.ylabel('Diagnosis')
plt.show()

# top 10 diagnoses by frequency using '.nlargest()'
top_diagnoses = data['Diagnosis'].value_counts().nlargest(10).index

# Filtering the dataset for only these top diagnoses
data_top_diagnoses = data[data['Diagnosis'].isin(top_diagnoses)]

# Count plot for the top 10 diagnoses
plt.figure(figsize=(12, 8))
sns.countplot(y='Diagnosis', data=data_top_diagnoses, order=data_top_diagnoses['Diagnosis'].value_counts().index)
plt.title('Top 10 Diagnoses Distribution')
plt.xlabel('Count')
plt.ylabel('Diagnosis')
plt.show()

# Analyzing diagnosis frequencies
diagnosis_counts = data['Diagnosis'].value_counts()
diagnosis_percentages = data['Diagnosis'].value_counts(normalize=True) * 100

diagnosis_summary = pd.DataFrame({'Count': diagnosis_counts, 'Percentage': diagnosis_percentages})
print(diagnosis_summary)

# Saving diagnosis summary to CSV using '.to_csv'
diagnosis_summary.to_csv('diagnosis_summary.csv', index=True)

# Count plot for Indication
plt.figure(figsize=(12, 8))
sns.countplot(y='Indication', data=data, order=data['Indication'].value_counts().index)
plt.title('Distribution of Indications')
plt.xlabel('Count')
plt.ylabel('Indication')
plt.show()

# Analyzing indication frequencies
indication_counts = data['Indication'].value_counts()
indication_percentages = data['Indication'].value_counts(normalize=True) * 100

indication_summary = pd.DataFrame({'Count': indication_counts, 'Percentage': indication_percentages})
print(indication_summary)

# Saving indication summary to CSV
indication_summary.to_csv('indication_summary.csv', index=True)

# Top 10 indications by frequency using '.nlargest()'
top_indications = data['Indication'].value_counts().nlargest(10).index

# Filtering the dataset for only these top indications
data_top_indications = data[data['Indication'].isin(top_indications)]

# Count plot for the top 10 indications
plt.figure(figsize=(12, 8))
sns.countplot(y='Indication', data=data_top_indications, order=data_top_indications['Indication'].value_counts().index)
plt.title('Top 10 Indications Distribution')
plt.xlabel('Count')
plt.ylabel('Indication')
plt.show()

# Count plot for Name of Drug
plt.figure(figsize=(12, 8))
sns.countplot(y='Name of Drug', data=data, order=data['Name of Drug'].value_counts().index)
plt.title('Distribution of Drugs')
plt.xlabel('Count')
plt.ylabel('Name of Drug')
plt.show()

# Analyzing drug frequencies
drug_counts = data['Name of Drug'].value_counts()
drug_percentages = data['Name of Drug'].value_counts(normalize=True) * 100

drug_summary = pd.DataFrame({'Count': drug_counts, 'Percentage': drug_percentages})
print(drug_summary)

# Saving drug summary to CSV
drug_summary.to_csv('drug_summary.csv', index=True)

# Top 10 drugs by frequency
top_drugs = data['Name of Drug'].value_counts().nlargest(10).index

# Filtering the dataset for only these top drugs
data_top_drugs = data[data['Name of Drug'].isin(top_drugs)]

# Count plot for the top 10 drugs
plt.figure(figsize=(12, 8))
sns.countplot(y='Name of Drug', data=data_top_drugs, order=data_top_drugs['Name of Drug'].value_counts().index)
plt.title('Top 10 Drugs Distribution')
plt.xlabel('Count')
plt.ylabel('Name of Drug')
plt.show()

# Pivot Table for Sum of Dosage by Top 10 Drugs and Top 10 Indications
# Filter the dataset for only these top drugs and indications
data_top = data[data['Name of Drug'].isin(top_drugs) & data['Indication'].isin(top_indications)]

# Creating a pivot table for sum of dosage by Top 10 Drugs and Top 10 Indications using '.pivot_table'
pivot_sum_dosage_top_drugs_indications = data_top.pivot_table(index='Name of Drug', columns='Indication',
                                                              values='Dosage (gram)', aggfunc='sum')

# Printing the pivot table
print("Pivot Table: Sum of Dosage (gram) by Top 10 Drugs and Top 10 Indications")
print(pivot_sum_dosage_top_drugs_indications)

# Saving pivot table to CSV
pivot_sum_dosage_top_drugs_indications.to_csv('pivot_sum_dosage_top_drugs_indications.csv', index=True)
