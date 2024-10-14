import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Save the model to a file
joblib.dump(rf, '/home/jiaxi/cs525/Experiment/feature_extraction/rf/random_forest_model.joblib')

