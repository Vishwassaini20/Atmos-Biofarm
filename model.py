# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 2: Upload dataset directly from PC
# This will prompt you to select a file from your computer
from google.colab import files
uploaded = files.upload()

# Assuming the file is named 'crop_recommendation.csv'
# Step 3: Load the dataset
df = pd.read_csv(list(uploaded.keys())[0])  # Automatically uses the uploaded filename
print("Dataset Loaded Successfully")
df.head()

# Step 4: Explore the dataset (optional but recommended)

print(df.info()) 
print(df.describe())  
print(df.isnull().sum())  

# Step 5: Separate features and target variable
X = df.drop(columns=['label'])  # Replace 'label' with the actual target column name if different
y = df['label']

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data Split into Train and Test Sets")

# Step 7: Initialize and train the model
# Using RandomForestClassifier as an example
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Training Complete")

# Step 8: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Save the model
joblib.dump(model, 'farmbuddy_crop_model.pkl')
print("Model Saved as 'farmbuddy_crop_model.pkl'")

