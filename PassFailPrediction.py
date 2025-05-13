import pandas as pd
import numpy as np
import opendatasets as od
s=od.download("www.kaggle.com/datasets/whenamancodes/students-performance-in-exams")
data = pd.read_csv("/content/students-performance-in-exams/exams.csv")
data.info()
data.drop(["gender", "race/ethnicity", "lunch"], axis=1, inplace=True, errors='ignore')
data.info()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Preprocessing: Convert categorical 'parental level of education' and 'test preparation course' to numerical
le_education = LabelEncoder()
data['parental level of education'] = le_education.fit_transform(data['parental level of education'])

le_test = LabelEncoder()
data['test preparation course'] = le_test.fit_transform(data['test preparation course'])

# Define features (X) and target (y)
features = ['parental level of education', 'test preparation course', 'writing score'] # Using writing score as a feature too
target = 'math score' # Let's try to predict math score

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# The model is trained, and its accuracy is evaluated using R-squared.
# The training and evaluation are completed.

# prompt: gete the data from user and predit the completion

# Function to get user input and predict completion
def predict_completion_from_user_input(model, data_columns):
    print("Please enter the following information:")

    user_data = {}
    # Iterate through the expected columns (excluding the target 'math score')
    # and prompt the user for input for the original categorical features
    original_categorical_features = [
        'race/ethnicity',
        'parental level of education',
        'lunch',
        'test preparation course'
    ]

    for feature in original_categorical_features:
        user_input = input(f"Enter {feature}: ")
        user_data[feature] = user_input

    # Create a pandas DataFrame from user input
    user_df = pd.DataFrame([user_data])

    # Apply the same one-hot encoding as used for training data
    user_df = pd.get_dummies(user_df, columns=original_categorical_features)

    # Ensure the user DataFrame has the same columns as the training data
    # Add missing columns filled with 0 (for categories not present in user input)
    # and reindex to match the order of the training data
    missing_cols = set(data_columns) - set(user_df.columns)
    for c in missing_cols:
        user_df[c] = 0
    user_df = user_df[data_columns]

    # Predict the completion (pass/fail)
    prediction = model.predict(user_df)

    if prediction[0] == 1:
        print("Prediction: The student is likely to pass.")
    else:
        print("Prediction: The student is likely to fail.")

# Get the list of columns used for training (excluding the target)
# We need to get the columns from the training data before it was split
# So we use the columns from the original 'X' DataFrame
training_columns = X.columns

# Get data from user and predict
predict_completion_from_user_input(model, training_columns)
