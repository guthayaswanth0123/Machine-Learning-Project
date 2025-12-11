# BMI Category Prediction Using Machine Learning

This project predicts **BMI Categories** (Underweight, Normal, Overweight, Obese) based on various lifestyle and health features collected from a survey dataset.  
Multiple machine learning models were implemented and compared to identify the most effective classifier.

---

## 🧠 Project Overview

The dataset includes features such as:
- Age  
- Gender  
- Height & Weight  
- Calculated BMI  
- Sleep duration  
- Smoking, Alcohol consumption  
- Exercise habits  
- Diet type  
- Family history of diabetes  

After preprocessing (cleaning, encoding, scaling, outlier handling), four supervised learning algorithms were applied.

---

## 📌 Machine Learning Algorithms Used

### 1️⃣ Logistic Regression  
A linear model used for multi-class classification using the **One-vs-Rest strategy**.  
- Simple and interpretable  
- Works well with scaled numerical features  
- Baseline model for comparison  

**Advantages:**  
✔ Fast, efficient, explainable  
✔ Good benchmark model  

**Limitations:**  
✖ Not suitable for complex non-linear patterns  

---

### 2️⃣ K-Nearest Neighbors (KNN)  
A non-parametric model that classifies a sample based on its **K closest neighbors**.  
- Distance-based classification  
- Works well for smaller datasets  

**Advantages:**  
✔ Easy to implement  
✔ Naturally handles multi-class classification  

**Limitations:**  
✖ Slow for large datasets  
✖ Sensitive to scaling (needs StandardScaler)  

---

### 3️⃣ Support Vector Machine (SVM)  
Implemented with **RBF Kernel** to capture non-linear decision boundaries.  
- Powerful and high-performance model  
- Can handle complex relationships  

**Advantages:**  
✔ Excellent accuracy with proper scaling  
✔ Works well in high-dimensional space  

**Limitations:**  
✖ Computationally expensive  
✖ Requires parameter tuning  

---

### 4️⃣ Decision Tree Classifier  
A tree-based algorithm that splits data using **Gini impurity**.  
- Highly interpretable  
- Provides feature importance ranking  

**Advantages:**  
✔ Easy to understand  
✔ No scaling required  
✔ Shows which features impact prediction the most  

**Limitations:**  
✖ Can overfit (needs pruning or ensembles)  

---

## 📊 Visualizations Included

- Confusion Matrix Heatmaps  
- ROC Curves (Multi-class)  
- Feature Importance (Decision Tree)  
- Actual vs Predicted Category Comparison  
- Decision Tree Structure Visualization  

These plots help interpret model performance and decision boundaries.

---

## 🏆 Model Comparison (Summary)

| Algorithm                    | Accuracy | Strengths                                   | Weaknesses                                    | 
|--------------------          |----------|---------------------------------------------|-----------------------------------------------|
| Logistic Regression          | **96%**  | Fast, interpretable, good baseline model    | Struggles with non-linear data                |
| K-Nearest Neighbors (KNN)    | **76%**  | Simple, intuitive, no training time         | Slow for large datasets, sensitive to scaling |
| Support Vector Machine (SVM) | **93%**  | Very accurate, handles non-linear patterns  | Computationally expensive                     |
| Decision Tree                | **99%**  | Shows feature importance, easy to interpret | Can overfit without pruning                   |




