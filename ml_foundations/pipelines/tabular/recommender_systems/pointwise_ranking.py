import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data (query, document features, and relevance labels)
# Replace this with your actual data
queries = np.random.randint(1, 11, size=100)  # Query IDs
document_features = np.random.rand(100, 5)  # 100 documents with 5 features each
labels = np.random.randint(1, 6, size=100)  # Relevance scores between 1 and 5

# Combine query and document features
features = np.column_stack((queries, document_features))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Example: Make predictions for new query and document data
new_query = np.array([1])  # New query ID
new_document_features = np.random.rand(5).reshape(1, -1)  # New document features
new_data = np.column_stack((new_query, new_document_features))
predicted_score = model.predict(new_data)
print("Predicted Score for New Data:", predicted_score[0])