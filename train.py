# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# 1. Import Dataset


df = pd.read_csv("Salary_Data.csv")



# Part A - Data Understanding


print("First Five Records:")
print(df.head())

print("\nDataset Dimensions:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())



# Part B - Exploratory Data Analysis


# 1. Salary Distribution

plt.figure(figsize=(7,5))
plt.hist(df["Salary"], bins=20, color="blue")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Salary Distribution")
plt.show()


# Observation:
# Salary values are distributed across different ranges.
# Most employees fall within the common salary range.



# 2. Experience vs Salary

plt.figure(figsize=(7,5))
plt.scatter(
    df["Years of Experience"],
    df["Salary"],
    color="green"
)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()


# Observation:
# Salary generally increases with experience.
# Experienced employees receive higher packages.



# 3. Education Level Count

education_count = df["Education Level"].value_counts()

plt.figure(figsize=(7,5))
plt.bar(
    education_count.index,
    education_count.values,
    color="orange"
)

plt.xlabel("Education Level")
plt.ylabel("Count")
plt.title("Education Level Distribution")

plt.xticks(rotation=45)
plt.show()


# Observation:
# Different education groups are present in the dataset.
# Some education levels have more candidates than others.



# 4. Salary by Job Title

job_salary = df.groupby("Job Title")["Salary"].mean()

plt.figure(figsize=(10,5))

plt.bar(
    job_salary.index,
    job_salary.values,
    color="purple"
)

plt.xlabel("Job Title")
plt.ylabel("Average Salary")
plt.title("Average Salary by Job Title")

plt.xticks(rotation=90)
plt.show()


# Observation:
# Salary varies according to job role.
# Technical roles may have higher average salaries.



# 5. Correlation Heatmap using Matplotlib

numeric_data = df.select_dtypes(
    include=np.number
)

correlation = numeric_data.corr()


plt.figure(figsize=(7,5))

plt.imshow(
    correlation,
    cmap="coolwarm"
)

plt.colorbar()


plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)


for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(
            j,
            i,
            round(correlation.iloc[i,j],2),
            ha="center",
            va="center"
        )


plt.title("Correlation Heatmap")
plt.show()

# 6. Boxplot for Salary

plt.figure(figsize=(7,5))

plt.boxplot(
    df["Salary"],
    patch_artist=True,
    boxprops=dict(facecolor="skyblue")
)

plt.ylabel("Salary")
plt.title("Salary Boxplot")

plt.show()

# --------------------------------
# Part C - Data Preprocessing
# --------------------------------


# Handle Missing Values

df.fillna(method="ffill", inplace=True)



# Encode Categorical Columns

encoders = {}


categorical_columns = [
    "Gender",
    "Education Level",
    "Job Title"
]


for col in categorical_columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le



# Input and Target

X = df.drop("Salary", axis=1)

y = df["Salary"]



# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# --------------------------------
# Part D - Model Building
# --------------------------------


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train Model

model.fit(
    X_train,
    y_train
)



# Prediction

y_pred = model.predict(X_test)



# Evaluation

print("\nModel Performance")

print(
    "R2 Score:",
    r2_score(y_test,y_pred)
)

print(
    "MAE:",
    mean_absolute_error(y_test,y_pred)
)

print(
    "MSE:",
    mean_squared_error(y_test,y_pred)
)



# Display Predictions

prediction = pd.DataFrame(
    {
        "Actual Salary": y_test,
        "Predicted Salary": y_pred
    }
)

print("\nPredicted Salary:")
print(prediction.head(10))



# --------------------------------
# Save Model
# --------------------------------

joblib.dump(
    model,
    "salary_model.pkl"
)

joblib.dump(
    encoders,
    "encoders.pkl"
)    


joblib.dump(
    list(X.columns),
    "features.pkl"
)


print("\nModel Saved Successfully")