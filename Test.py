TESTING THE LOGISTIC REGRESSION APPROACH

# Assume you still have w, b, train_means, train_stds from training
df=pd.read_csv("/content/loan-test.csv")
# Drop ID and Gender just like training
df.drop(["Loan_ID", "Gender"], axis=1, inplace=True)

# Encode same way as training
df["Education"] = df.Education.map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df.Self_Employed.map({"Yes": 1, "No": 0})
df["Married"] = df.Married.map({"Yes": 1, "No": 0})
df["Property_Area"] = df.Property_Area.map({"Urban": 2, "Semiurban": 1, "Rural": 0})

# Dependents: convert "3+" to 3
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(float)

# Drop rows with missing values (or handle them)
df = df.dropna()

# Normalize using training mean/std
cols_to_normalize = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
df[cols_to_normalize] = (df[cols_to_normalize] - trains_mean) / trains_std

# Ensure column order matches training
X_test = df.astype(float).values

# Predict
def predict(d):
    y_prob = sigmoid(np.dot(d, w) + b)
    return (y_prob > 0.5).astype(int)

predictions = predict(X_test)
print("Predictions:", predictions)


import pandas as pd
import numpy as np

# Example single row of new data
new_data = pd.DataFrame([{
    "Loan_ID": "LP001008",
    "Gender": "Male",
    "Married": "No",
    "Dependents": 0,
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5849,
    "CoapplicantIncome": 0,
    "LoanAmount": 60,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
}])

# WITH UNKNOWN DATA
# --- Apply SAME preprocessing as training ---
new_data.drop(["Loan_ID", "Gender"], axis=1, inplace=True)

new_data["Education"] = new_data.Education.map({"Graduate": 1, "Not Graduate": 0})
new_data["Self_Employed"] = new_data.Self_Employed.map({"Yes": 1, "No": 0})
new_data["Married"] = new_data.Married.map({"Yes": 1, "No": 0})
new_data["Property_Area"] = new_data.Property_Area.map({"Urban": 2, "Semiurban": 1, "Rural": 0})

new_data["Dependents"] = pd.to_numeric(new_data["Dependents"], errors="coerce")

# Normalize with training mean/std (must come from training data)
cols_to_normalize = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
new_data[cols_to_normalize] = (new_data[cols_to_normalize] - trains_mean) / trains_std

# Convert to numpy array for prediction
X_single = new_data.astype(float).values

# Prediction
pred = predict(X_single)  # using your trained w, b
print("Prediction:", pred[0])
