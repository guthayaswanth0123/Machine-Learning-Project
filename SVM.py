import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

df = pd.read_csv(r"C:\Users\nithe\OneDrive\Documents\Assessment.csv")
df = df.drop(columns=["Name"])
df = df.drop_duplicates()

df["BMI"] = df["Weight (in kg)"] / ((df["Height (in cm)"] / 100) ** 2)

import re
def extract_sleep_int(x):
    x = str(x)
    nums = re.findall(r'\d+', x)
    if len(nums) == 2:
        return (int(nums[0]) + int(nums[1])) // 2
    if len(nums) == 1:
        return int(nums[0])
    if "less" in x.lower():
        return 5
    if "more" in x.lower():
        return 9
    return 7

df["Sleep_Hours"] = df["How many hours do you sleep daily?"].apply(extract_sleep_int)
df = df.drop(columns=["How many hours do you sleep daily?"])

num_cols = ["Age", "Height (in cm)", "Weight (in kg)", "BMI"]
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))

for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

df["Do you smoke?"] = df["Do you smoke?"].str.strip().replace("NO", "No")
df["Do you smoke?"] = df["Do you smoke?"].map({"Yes":1, "No":0, "Occasionally":2})
df["Do you consume alcohol?"] = df["Do you consume alcohol?"].map({"Never":0, "Occasionally":1, "Frequently":2})
df["Do you exercise daily?"] = df["Do you exercise daily?"].map({"Yes":1, "No":0, "Sometimes":2})
df["Does your family have a history of diabetes?"] = df["Does your family have a history of diabetes?"].map({"Yes":1, "No":0, "Not Sure":2})
df["Gender"] = df["Gender"].map({"Male":0, "Female":1, "Prefer not to say":2, "Other":3})
df["What is your diet type?"] = df["What is your diet type?"].map({"Vegetarian":0, "Non-Vegetarian":1, "Mixed":2})

def bmi_cat(bmi):
    if bmi < 18.5: return "Underweight Condition"
    elif bmi < 25: return "Normal & Healthy Weight"
    elif bmi < 30: return "Risk of Overweight"
    else: return "High Obesity Risk"

df["BMI_Category"] = df["BMI"].apply(bmi_cat)

df["BMI_Category"] = df["BMI_Category"].map({
    "Underweight Condition": 0,
    "Normal & Healthy Weight": 1,
    "Risk of Overweight": 2,
    "High Obesity Risk": 3
})

df = df.drop(columns=["Are you healthy?"])

X = df.drop(columns=["BMI_Category"])
y = df["BMI_Category"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
num_cols = X_train.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("\nClassification Report:\n", classification_report(y_test, svm_pred))



#VISULIZATION


# CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, svm_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Underweight", "Normal", "Overweight", "Obese"],
            yticklabels=["Underweight", "Normal", "Overweight", "Obese"])

plt.title("Confusion Matrix – SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# TRAIN SVM WITH PROBABILITIES
svm_roc = SVC(kernel="rbf", probability=True)
svm_roc.fit(X_train, y_train)

y_score = svm_roc.predict_proba(X_test)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# BINARIZE TARGET
y_test_bin = label_binarize(y_test, classes=[0,1,2,3])
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(8,6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Class {i} ROC (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], 'k--')

plt.title("Multi-Class ROC Curve – SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

