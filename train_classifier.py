import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels from the dictionary
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check the shape of the data
print(f"Original data shape: {data.shape}")

# Flatten the data if it has more than 2 dimensions
if data.ndim > 2:
    data = data.reshape(data.shape[0], -1)  # Reshape to (num_samples, num_features)

# Check the new shape
print(f"Flattened data shape: {data.shape}")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define the base model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],           # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],           # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],           # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]              # Minimum number of samples required to be at a leaf node
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Retrieve the best model from GridSearchCV
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Train the optimized model
best_model.fit(x_train, y_train)

# Make predictions using the test set
y_predict = best_model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Save the optimized model
with open('optimized_model.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_predict)

# Visualize the Confusion Matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
