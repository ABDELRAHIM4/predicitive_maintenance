import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the dataset
data = pd.read_csv("predictive_maintenance.csv")
# Display the first few rows of the dataset
data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})
data['Failure Type'] = data['Failure Type'].map({'No Failure': 0, 'Sort descending': 1, 'Heat Dissipation Failure': 2,
                                     'Power Failure': 3, 'Overstrain Failure': 4, 'Tool Wear Failure': 5})
data = data.dropna()

print(data.head())
# Split the dataset into features and target variable
X = data.drop('Target', axis=1)
y = data['Failure Type']
X = pd.get_dummies(X, columns=['Product ID'], drop_first=True)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a logistic regression model
model = LogisticRegression()
# Train the model
model.fit(X_train, y_train)
# Make predictions on the test set
predictions = model.predict(X_test)
# Display the predictions
print("Predictions on the test set:")
print(predictions)
# Evaluate the model's performance
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")