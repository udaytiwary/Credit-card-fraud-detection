#!/usr/bin/env python
# coding: utf-8
projectCredit Card Fraud Detection
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


# In[3]:


# Load dataset
DATASET_PATH = "creditcard.csv"
df = pd.read_csv(DATASET_PATH)


# In[5]:


df


# In[7]:


# Display basic information about the dataset
print("Dataset Overview:")
print(df.info())


# In[9]:


# Display first few rows
print("First 5 rows:")
print(df.head())


# In[11]:


# Check for missing values
print("Missing Values:")
print(df.isnull().sum())


# In[13]:


# Display class distribution
print("Class Distribution:")
print(df['Class'].value_counts())
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()


# In[15]:


# Data Preprocessing
# Check for required columns
required_columns = ['Time', 'Amount']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Warning: The dataset is missing the following expected columns: {missing_columns}")
else:
    print("Dataset contains 'Time' and 'Amount' columns.")


# In[17]:


# Address class imbalance using SMOTE
X = df.drop(columns=['Class'])
y = df['Class']


# In[19]:


X


# In[21]:


y


# In[23]:


# Splitting data before applying SMOTE to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[25]:


print("Class distribution before SMOTE:", Counter(y_train))
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", Counter(y_train_resampled))


# In[27]:


# Feature Engineering
print("Engineering new features...")


# In[29]:


# Transaction frequency
X_train_resampled['Transaction_Frequency'] = X_train_resampled.groupby('Time')['Amount'].transform('count')
X_test['Transaction_Frequency'] = X_test.groupby('Time')['Amount'].transform('count')


# In[31]:


# Spending patterns (Rolling Mean over last 10 transactions)
X_train_resampled['Spending_Pattern'] = X_train_resampled['Amount'].rolling(window=10, min_periods=1).mean()
X_test['Spending_Pattern'] = X_test['Amount'].rolling(window=10, min_periods=1).mean()


# In[33]:


print("Feature Engineering completed!")


# In[35]:


# Feature Scaling
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)


# In[37]:


# Train Logistic Regression Model
print("Training Logistic Regression Model...")
log_model = LogisticRegression()
log_model.fit(X_train_resampled, y_train_resampled)


# In[39]:


# Predictions using Logistic Regression
y_log_pred = log_model.predict(X_test)


# In[41]:


# Train Random Forest Model with Hyperparameter Tuning
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)


# In[43]:


# Predictions using Random Forest
y_rf_pred = rf_model.predict(X_test)


# In[45]:


# Model Evaluation
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_log_pred))
print("Classification Report:\n", classification_report(y_test, y_log_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_log_pred))


# In[47]:


print("Random Forest Performance:")
print("Accuracy:", accuracy_score(y_test, y_rf_pred))
print("Classification Report:\n", classification_report(y_test, y_rf_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_rf_pred))


# In[ ]:




