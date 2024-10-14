from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import train_test_split
import joblib

# Load the Iris dataset
iris = datasets.load_iris()

# Use one feature (e.g., Sepal length) as the independent variable
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, '/home/jiaxi/cs525/Experiment/feature_extraction/lr/linear_regression_model.pkl')
