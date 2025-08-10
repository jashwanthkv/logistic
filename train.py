TRAINING ( FINDIGN BEST WEIGHT AND BIAS)
import pandas as pd
import numpy as np

# Load and clean
df = pd.read_csv("/content/loan-train.csv")
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Drop irrelevant columns
df.drop(["Loan_ID", "Gender"], axis=1, inplace=True)

# Encode categorical features
df["Education"] = df.Education.map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df.Self_Employed.map({"Yes": 1, "No": 0})
df["Married"] = df.Married.map({"Yes": 1, "No": 0})
df["Property_Area"] = df.Property_Area.map({"Urban": 2, "Semiurban": 1, "Rural": 0})

# Convert Dependents to numeric (remove '+')
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(float)

# Remove missing values
df = df.dropna()

# Normalize continuous features
cols_to_normalize = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
df[cols_to_normalize] = (df[cols_to_normalize] - df[cols_to_normalize].mean()) / df[cols_to_normalize].std()
trains_mean=df[cols_to_normalize].mean()
trains_std=df[cols_to_normalize].std()
# Features and target
X = df.drop("Loan_Status", axis=1).astype(float).values
y = df["Loan_Status"].values

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression from scratch
def regression(x, y):
    n_sam, n_features = x.shape
    epochs = 5000
    weight = np.zeros(n_features)
    bias = 0
    lr = 0.01
    
    for _ in range(epochs):
        z = np.dot(x, weight) + bias
        y_pred = sigmoid(z)
        error = y_pred - y
        dw = (1/n_sam) * np.dot(x.T, error)
        db = (1/n_sam) * np.sum(error)
        weight -= lr * dw
        bias -= lr * db
    
    return weight, bias

# Train the model
w, b = regression(X, y)

# Prediction function
def predict(d):
    y_prob = sigmoid(np.dot(d, w) + b)
    return (y_prob>0.5).astype(int)

# Accuracy
preds = predict(X)
accuracy = np.mean(preds == y)
print("Accuracy:", round(accuracy, 4))
print("Weights:", w)
print("Bias:", b)
