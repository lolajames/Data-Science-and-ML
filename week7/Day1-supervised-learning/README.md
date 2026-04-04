
# Lab Title
## Supervised Learning Models for Customer Churn Prediction


# Objective

The objective of this lab is to reproduce classroom exercises demonstrating the fundamentals of **Supervised Learning** using a **customer churn prediction** problem.

This lab focuses on:

- Preprocessing data
- Label encoding
- Scaling
- Splitting data into training and validation sets
- Training machine learning models
- Evaluating model performance
- Implementing models using both **OOP Approach** and **Procedural Approach**
- Saving trained models

---


This lab helps build foundational skills required for **classification tasks, predictive modeling, and customer churn analysis in Machine Learning**.

---

# Problem Statement

Develop a predictive model to identify customers at risk of churning from an investment bank, enabling proactive retention strategies to minimize customer loss and maximize revenue growth.

---

# About the Dataset

There are **14 columns/features** and **10,000 rows/samples**.

- **RowNumber** — corresponds to the record (row) number and has no effect on the output.
- **CustomerId** — contains random values and has no effect on customer leaving the bank.
- **Surname** — the surname of a customer has no impact on their decision to leave the bank.
- **CreditScore** — can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
- **Geography** — a customer’s location can affect their decision to leave the bank.
- **Gender** — it’s interesting to explore whether gender plays a role in a customer leaving the bank.
- **Age** — this is certainly relevant, since older customers are less likely to leave their bank than younger ones.
- **Tenure** — refers to the number of years that the customer has been a client of the bank.
- **Balance** — a very good indicator of customer churn.
- **NumOfProducts** — refers to the number of products that a customer has purchased through the bank.
- **HasCrCard** — denotes whether or not a customer has a credit card.
- **IsActiveMember** — active customers are less likely to leave the bank.
- **EstimatedSalary** — people with lower salaries are more likely to leave the bank compared to those with higher salaries.
- **Exited** — whether or not the customer left the bank.

---

# Tools Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Joblib**
- **Jupyter Notebook / Google Colab**
- **GitHub**

---

# Step-by-Step Process

## Step 1 – Import Required Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score
from sklearn.svm import SVC
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
````

---

## Step 2 – Load Data

```python
# Load data
data = pd.read_excel('churn.xlsx')
```

```python
data.head()
```

```python
data.tail()
```

```python
data['Age'].unique()
```

```python
data.info()
```

```python
# is null?
isnull = data.isnull().sum()
isnull
```

---

## Step 3 – Preprocess Data

```python
# Preprocess data
selected_features = [
    'CreditScore', 'Geography', 'Gender', 'Age',
    'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary'
]
X = data[selected_features]
y = data[['Exited']]
```

---

## Step 4 – Label Encoding

Categorical variables need to be encoded into numerical format before training machine learning models.

```python
# Label encoding
le = LabelEncoder()
X['Geography'] = le.fit_transform(X['Geography'])
X['Gender'] = le.fit_transform(X['Gender'])
```

```python
X['Gender']
```

---

## Step 5 – Scaling

Scaling standardizes the range of selected numerical features.

```python
# Scaling
scaler = MinMaxScaler()
X[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']] = scaler.fit_transform(X[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']])
```

```python
X['Age'].unique()
```

---

## Step 6 – Split Data

```python
# Split data
train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=0, train_size=0.8
)
```

---

# K-Nearest Neighbors (KNN)

The K-Nearest Neighbors (KNN) algorithm is a simple and effective machine learning technique that classifies data points by finding the K most similar instances to a new input and voting for the target class or value. 

## Commonly Used Hyperparameters for KNN

* **n_neighbors**: 3, 5, 10, 20
* **weights**: `'uniform'`, `'distance'`
* **algorithm**: `'brute'`, `'kd_tree'`, `'ball_tree'`
* **leaf_size**: 10, 20, 30
* **p**: 1, 2
* **metric**: `'minkowski'`, `'euclidean'`, `'manhattan'`, `'chebyshev'`

---

## KNN – Train Model

```python
# Train model
kneighbor = KNeighborsClassifier(n_neighbors=2, metric='euclidean', weights='uniform', algorithm='auto', leaf_size=50, p=2)
kneighbor.fit(train_X, train_y)
```

## KNN – Evaluate Model

```python
# Evaluate model
val_prediction = kneighbor.predict(val_X)
y_pred_proba = kneighbor.predict_proba(val_X)[:,1]
accuracy = accuracy_score(val_y, val_prediction)
print(f'Model accuracy: {accuracy}')
```

```python
print(confusion_matrix(val_y, val_prediction))
print(classification_report(val_y, val_prediction))
```

```python
auc = roc_auc_score(val_y, y_pred_proba)
print(auc)
```

## KNN – Save Model

```python
# Save model
joblib.dump(kneighbor, 'churn_model.pkl')
```

---

## KNN – OOP Approach

```python
class ChurnPrediction:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.train_X = None
        self.val_X = None
        self.train_y = None
        self.val_y = None
        self.model = None

    def load_data(self):
        self.data = pd.read_excel(self.file_path)

    def preprocess_data(self):
        selected_features = [
            'CreditScore', 'Geography', 'Gender', 'Age',
            'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'EstimatedSalary'
        ]
        self.X = self.data[selected_features]
        self.y = self.data[['Exited']]

        # Encoding categorical variables
        le = LabelEncoder()
        self.X['Geography'] = le.fit_transform(self.X['Geography'])
        self.X['Gender'] = le.fit_transform(self.X['Gender'])

        # Scaling numerical variables
        scaler = MinMaxScaler()
        self.X[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']] = scaler.fit_transform(self.X[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']])

    def split_data(self):
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.X, self.y, random_state=0, train_size=0.8
        )

    def train_model(self):
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.train_X, self.train_y)

    def evaluate_model(self):
        val_prediction = self.model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, val_prediction)
        print(f'Model accuracy: {accuracy}')
        y_pred_proba = self.model.predict_proba(self.val_X)[:,1]
        auc = roc_auc_score(self.val_y, y_pred_proba)
        print(f'Model auc score: {auc}')
        return accuracy, auc

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
```

### KNN – OOP Approach # Usage

```python
# Usage
churn = ChurnPrediction('churn.xlsx')
churn.load_data()
churn.preprocess_data()
churn.split_data()
churn.train_model()
accuracy = churn.evaluate_model()

# Save the model
churn.save_model('churn1_model.pkl')
```

---

## KNN – Procedural Approach

```python
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

def preprocess_data(data):
    selected_features = [
        'CreditScore', 'Geography', 'Gender', 'Age',
        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary'
    ]
    X = data[selected_features]
    y = data[['Exited']]

    # Label encoding
    le = LabelEncoder()
    X['Geography'] = le.fit_transform(X['Geography'])
    X['Gender'] = le.fit_transform(X['Gender'])

    # Scaling
    scaler = MinMaxScaler()
    X[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']] = scaler.fit_transform(X[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']])

    return X, y

def split_data(X, y):
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, random_state=0, train_size=0.8
    )
    return train_X, val_X, train_y, val_y

def train_model(train_X, train_y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_X, train_y)
    return model

def evaluate_model(model, val_X, val_y):
    val_prediction = model.predict(val_X)
    accuracy = accuracy_score(val_y, val_prediction)
    print(f'Model accuracy: {accuracy}')

    auc = roc_auc_score(val_y, val_prediction)
    print(f'Model auc score: {auc}')
    return accuracy, auc

def save_model(model, model_path):
    joblib.dump(model, model_path)

def load_model(model_path):
    model = joblib.load(model_path)
    return model
```

### KNN – Procedural Approach # Usage

```python
# Usage
file_path = 'churn.xlsx'
data = load_data(file_path)
X, y = preprocess_data(data)
train_X, val_X, train_y, val_y = split_data(X, y)
model = train_model(train_X, train_y)
accuracy, auc = evaluate_model(model, val_X, val_y)
save_model(model, 'churn2_model.pkl')
```

---

# Decision Tree Classifier

A Decision Tree Classifier is a type of supervised learning algorithm in machine learning. It works by creating a tree-like model of decisions and their possible consequences. 

## Commonly Used Hyperparameters for Decision Tree Classifier

* **criterion**: `'gini'`, `'entropy'`
* **max_depth**: 3, 5, 10, None
* **min_samples_split**: 2, 5, 10
* **min_samples_leaf**: 1, 5, 10
* **max_features**: `'auto'`, `'sqrt'`, `'log2'`, None
* **random_state**: 0, 42, 100
* **class_weight**: `'balanced'`, `'balanced_subsample'`, None

---

## Decision Tree Classifier – OOP Approach

```python
class ChurnPrediction:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.train_X = None
        self.val_X = None
        self.train_y = None
        self.val_y = None
        self.model = None

    def load_data(self):
        self.data = pd.read_excel(self.file_path)

    def preprocess_data(self):
        selected_features = [
            'CreditScore', 'Geography', 'Gender', 'Age',
            'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'EstimatedSalary'
        ]
        self.X = self.data[selected_features]
        self.y = self.data[['Exited']]

        self.X = pd.get_dummies(self.X, columns = ["Geography", "Gender"])
        #self.X.drop(columns=["Geography", "Gender"],axis=1, inplace=True)

    def split_data(self):
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.X, self.y, random_state=0, train_size=0.8
        )

    def train_model(self):
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(self.train_X, self.train_y)

    def evaluate_model(self):
        val_prediction = self.model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, val_prediction)
        print(f'Model accuracy: {accuracy}')
        y_pred_proba = self.model.predict_proba(self.val_X)[:,1]
        auc = roc_auc_score(self.val_y, y_pred_proba)
        print(f'Model auc score: {auc}')
        return accuracy, auc

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
```

### Decision Tree Classifier – OOP Approach # Usage

```python
# Usage
churn = ChurnPrediction('churn.xlsx')
churn.load_data()
churn.preprocess_data()
churn.split_data()
churn.train_model()
accuracy, auc = churn.evaluate_model()

# Save the model
churn.save_model('churn3_model.pkl')
```

---

## Decision Tree Classifier – Procedural Approach

```python
def load_data(file_path):
    return pd.read_excel(file_path)

def preprocess_data(data):
    selected_features = [
        'CreditScore', 'Geography', 'Gender', 'Age',
        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary'
    ]
    X = data[selected_features]
    y = data[['Exited']]

    X = pd.get_dummies(X, columns = ["Geography", "Gender"])
    #X.drop(columns=["Geography", "Gender"], inplace=True)

    return X, y

def split_data(X, y):
    return train_test_split(
        X, y, random_state=0, train_size=0.8
    )

def train_model(X, y):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    val_prediction = model.predict(X)
    accuracy = accuracy_score(y, val_prediction)
    print(f'Model accuracy: {accuracy}')
    y_pred_proba = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, y_pred_proba)
    print(f'Model auc score: {auc}')
    return accuracy, auc

def save_model(model, model_path):
    joblib.dump(model, model_path)
```

### Decision Tree Classifier – Procedural Approach # Usage

```python
# Usage
file_path = 'churn.xlsx'
data = load_data(file_path)
X, y = preprocess_data(data)
train_X, val_X, train_y, val_y = split_data(X, y)
model = train_model(train_X, train_y)
accuracy, auc = evaluate_model(model, val_X, val_y)
save_model(model, 'churn_model.pkl')
```

---

# Random Forest Classifier

Random Forest is a supervised learning algorithm that combines multiple decision trees to produce a more accurate and stable prediction model. 

## The Most Commonly Used Hyperparameters for Random Forest Classifier

* **n_estimators**: 10, 50, 100, 200
* **criterion**: `'gini'`, `'entropy'`
* **max_depth**: 3, 5, 10, None
* **min_samples_split**: 2, 5, 10
* **min_samples_leaf**: 1, 5, 10
* **max_features**: `'auto'`, `'sqrt'`, `'log2'`, None
* **max_leaf_nodes**: 10, 50, 100, None
* **min_impurity_decrease**: 0.0, 0.1, 0.5
* **bootstrap**: True, False
* **oob_score**: True, False
* **random_state**: 0, 42, 100
* **class_weight**: `'balanced'`, `'balanced_subsample'`, None

---

## Random Forest Classifier – OOP Approach

```python
class ChurnPrediction:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.train_X = None
        self.val_X = None
        self.train_y = None
        self.val_y = None
        self.model = None

    def load_data(self):
        self.data = pd.read_excel(self.file_path)

    def preprocess_data(self):
        selected_features = [
            'CreditScore', 'Geography', 'Gender', 'Age',
            'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'EstimatedSalary'
        ]
        self.X = self.data[selected_features]
        self.y = self.data[['Exited']]

        self.X = pd.get_dummies(self.X, columns = ["Geography", "Gender"])

    def split_data(self):
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.X, self.y, random_state=0, train_size=0.8
        )

    def train_model(self):
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.train_X, self.train_y)

    def evaluate_model(self):
        val_prediction = self.model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, val_prediction)
        print(f'Model accuracy: {accuracy}')
        y_pred_proba = self.model.predict_proba(self.val_X)[:,1]
        auc = roc_auc_score(self.val_y, y_pred_proba)
        print(f'Model auc score: {auc}')
        return accuracy, auc

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
```

### Random Forest Classifier – OOP Approach # Usage

```python
# Usage
churn = ChurnPrediction('churn.xlsx')
churn.load_data()
churn.preprocess_data()
churn.split_data()
churn.train_model()
accuracy, auc = churn.evaluate_model()

# Save the model
churn.save_model('churn4_model.pkl')
```

---

## Random Forest Classifier – Procedural Approach

```python
def load_data(file_path):
    return pd.read_excel(file_path)

def preprocess_data(data):
    selected_features = [
        'CreditScore', 'Geography', 'Gender', 'Age',
        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary'
    ]
    X = data[selected_features]
    y = data[['Exited']]

    X = pd.get_dummies(X, columns = ["Geography", "Gender"])

    return X, y

def split_data(X, y):
    return train_test_split(
        X, y, random_state=0, train_size=0.8
    )

def train_model(X, y):
    model = RandomForestClassifier(random_state=0)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    val_prediction = model.predict(X)
    accuracy = accuracy_score(y, val_prediction)
    print(f'Model accuracy: {accuracy}')
    y_pred_proba = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, y_pred_proba)
    print(f'Model auc score: {auc}')
    return accuracy, auc

def save_model(model, model_path):
    joblib.dump(model, model_path)
```

### Random Forest Classifier – Procedural Approach # Usage

```python
# Usage
file_path = 'churn.xlsx'
data = load_data(file_path)
X, y = preprocess_data(data)
train_X, val_X, train_y, val_y = split_data(X, y)
model = train_model(train_X, train_y)
accuracy, auc = evaluate_model(model, val_X, val_y)
save_model(model, 'churn_model.pkl')
```




````markdown
# Support Vector Machine (SVM)

Support Vector Machine (SVM) is a supervised learning algorithm used for both classification and regression tasks. It works by finding the optimal hyperplane that separates data points into different classes.

---

## The Most Commonly Used Hyperparameters for Support Vector Machines (SVMs)

- **C**: 0.1, 1, 10, 100  
- **kernel**: 'linear', 'rbf', 'poly', 'sigmoid'  
- **gamma**: 'scale', 'auto', 0.1, 0.01, 0.001  
- **degree**: 2, 3, 4, 5  

---

## SVM – Train Model

```python
svm_model = SVC(probability=True)
svm_model.fit(train_X, train_y)

svm_pred = svm_model.predict(val_X)
````

---

## SVM – Evaluate Model

```python
accuracy = accuracy_score(val_y, svm_pred)
print(f'Model accuracy: {accuracy}')

y_pred_proba = svm_model.predict_proba(val_X)[:,1]
auc = roc_auc_score(val_y, y_pred_proba)
print(f'Model auc score: {auc}')
```

---

## SVM – Save Model

```python
joblib.dump(svm_model, 'churn_svm_model.pkl')
```

---

## SVM – OOP Approach

```python
class SVMModel:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = SVC(probability=True)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        pred = self.model.predict(X)
        accuracy = accuracy_score(y, pred)
        print(f'Model accuracy: {accuracy}')

        y_pred_proba = self.model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, y_pred_proba)
        print(f'Model auc score: {auc}')

    def save(self, path):
        joblib.dump(self.model, path)
```

---

## SVM – OOP Approach # Usage

```python
svm = SVMModel()
svm.train(train_X, train_y)
svm.evaluate(val_X, val_y)

svm.save('churn_svm_oop.pkl')
```

---

## SVM – Procedural Approach

```python
def train_svm(train_X, train_y):
    model = SVC(probability=True)
    model.fit(train_X, train_y)
    return model

def evaluate_svm(model, val_X, val_y):
    pred = model.predict(val_X)
    accuracy = accuracy_score(val_y, pred)
    print(f'Model accuracy: {accuracy}')

    y_pred_proba = model.predict_proba(val_X)[:,1]
    auc = roc_auc_score(val_y, y_pred_proba)
    print(f'Model auc score: {auc}')

def save_svm(model, path):
    joblib.dump(model, path)
```

---

## SVM – Procedural Approach # Usage

```python
svm_model = train_svm(train_X, train_y)
evaluate_svm(svm_model, val_X, val_y)

save_svm(svm_model, 'churn_svm_procedural.pkl')
```


# Screenshots of Results

## Dataset Preview and Information

<img width="516" height="289" alt="image" src="https://github.com/user-attachments/assets/1d3ce891-9c77-4e60-96bf-58b4bcd1b590" />

<img width="547" height="298" alt="image" src="https://github.com/user-attachments/assets/ccffb96e-bbe6-41cb-b4df-0a08fd1db758" />

<img width="542" height="322" alt="image" src="https://github.com/user-attachments/assets/10935e1c-8593-4879-829b-90ae33b2f566" />

<img width="574" height="324" alt="image" src="https://github.com/user-attachments/assets/a12f7aa1-d007-4885-9c2d-60d860efe83f" />
<img width="574" height="324" alt="image" src="https://github.com/user-attachments/assets/8031c360-5742-412a-96fb-6369b79781d9" />


## Scaling and Train-Test Split
<img width="604" height="225" alt="image" src="https://github.com/user-attachments/assets/c7341aa4-e455-4038-b8d1-4a25ce3f76b7" />

## KNN  Train model
<img width="469" height="332" alt="image" src="https://github.com/user-attachments/assets/c299a9ca-6c7f-465a-b514-564020bb450b" />

## OOP Approach
<img width="691" height="367" alt="image" src="https://github.com/user-attachments/assets/5622438c-6e61-4cab-91ad-3b828858c384" />
<img width="359" height="167" alt="image" src="https://github.com/user-attachments/assets/4fcec515-cc50-419f-99e0-b0d56f2d2f4d" />


## Procedural Approach
<img width="661" height="376" alt="image" src="https://github.com/user-attachments/assets/5bd57b7e-ff93-4532-a5f1-b730847f661d" />
<img width="395" height="94" alt="image" src="https://github.com/user-attachments/assets/3052b1ef-6c87-4b15-ae65-cc640b087fc4" />

## Random Forest Classifier
<img width="419" height="393" alt="image" src="https://github.com/user-attachments/assets/9475ffc1-8f5e-41fb-b912-288b799915fe" />

<img width="427" height="113" alt="image" src="https://github.com/user-attachments/assets/3f93ba8a-3019-4e8c-8108-ebcbfaf6d76a" />

<img width="446" height="414" alt="image" src="https://github.com/user-attachments/assets/274ec7e5-2f4f-4b6a-9a9a-80a552b8cb59" />

## Support Vector Machines (SVMs)

<img width="457" height="425" alt="image" src="https://github.com/user-attachments/assets/2a0f4162-26bc-4573-9fd7-a2d3fe673262" />

<img width="441" height="170" alt="image" src="https://github.com/user-attachments/assets/28c7ac09-3320-455c-b4fe-c9e58948619d" />



# Lessons Learned

From this lab, the following skills were gained:

* Data preprocessing for supervised learning
* Label encoding and feature scaling
* Splitting datasets into training and validation sets
* Training KNN, Decision Tree, and Random Forest models
* Implementing both OOP and procedural machine learning workflows
* Evaluating classification models using accuracy, confusion matrix, classification report, and AUC
* Saving trained models with Joblib




