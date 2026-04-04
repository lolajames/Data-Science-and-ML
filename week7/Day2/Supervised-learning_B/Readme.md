<img width="427" height="395" alt="image" src="https://github.com/user-attachments/assets/7ea4022b-02e4-4537-ab40-40e75eab8b0e" />
# Lab Title
## Supervised Learning (B) – Advanced Model Implementation with XGBoost and LightGBM

---

# Objective

The objective of this lab is to reproduce classroom exercises demonstrating advanced supervised learning workflows using structured machine learning pipelines.

The lab focuses on:

- Loading dataset from Google Drive
- Data preprocessing and feature selection
- One-hot encoding categorical variables
- Feature scaling using `StandardScaler`
- Splitting data into training and validation sets
- Training advanced machine learning models
- Hyperparameter tuning using `GridSearchCV`
- Hyperparameter tuning using `RandomizedSearchCV`
- Implementing Object-Oriented Programming (OOP) approach
- Implementing Procedural approach
- Saving trained models

This lab helps build foundational skills required for **advanced machine learning pipelines, model optimization, and predictive analytics**.

---

# Tools Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- SciPy
- TensorFlow
- Joblib
- Google Colab
- GitHub

---

# Step-by-Step Process

---

## Step 1 – Import Required Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score
from sklearn.svm import SVC
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import uniform, randint
import scipy.stats as stats
````

---

## Step 2 – Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Step 3 – Set File Path

```python
file_path_ = '/content/drive/My Drive/Churn Project/churn.csv'
```

---

# OOP Approach – XGBoost Classifier

## Code

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
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        selected_features = [
            'CreditScore', 'Geography', 'Gender', 'Age',
            'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'EstimatedSalary'
        ]
        self.X = self.data[selected_features]
        self.y = self.data[['Exited']]

        self.X = pd.get_dummies(self.X, columns=["Geography", "Gender"])

    def split_data(self):
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.X, self.y, random_state=0, train_size=0.8
        )

    def train_model(self):
        self.model = xgb.XGBClassifier(random_state=42)
        self.model.fit(self.train_X, self.train_y)

    def evaluate_model(self):
        val_prediction = self.model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, val_prediction)
        print(f'Model accuracy: {accuracy}')
        y_pred_proba = self.model.predict_proba(self.val_X)[:, 1]
        auc = roc_auc_score(self.val_y, y_pred_proba)
        print(f'Model AUC score: {auc}')
        return accuracy, auc

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

# Usage
churn = ChurnPrediction(file_path_)
churn.load_data()
churn.preprocess_data()
churn.split_data()
churn.train_model()
accuracy, auc = churn.evaluate_model()

# Save the model
churn.save_model('churn_model.pkl')
```

---

# Procedural Approach – XGBoost with GridSearchCV

## Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to preprocess data
def preprocess_data(data):
    selected_features = [
        'CreditScore', 'Geography', 'Gender', 'Age',
        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary'
    ]
    X = data[selected_features]
    y = data[['Exited']]

    X = pd.get_dummies(X, columns=["Geography", "Gender"])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Function to split data
def split_data(X, y):
    return train_test_split(X, y, random_state=0, train_size=0.8)

# Function to train XGBoost model
def train_model_xgb(train_X, train_y):
    model = xgb.XGBClassifier(random_state=0)
    param_grid = {
        'n_estimators': [100, 200,500],
        'learning_rate': [0.001,0.01, 0.1],
        'max_depth': [3, 5, 7,9],
    }
    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3)
    grid_search.fit(train_X, train_y.values.ravel())
    return grid_search.best_estimator_

# Function to evaluate model
def evaluate_model(model, val_X, val_y):
    val_prediction = model.predict(val_X)
    accuracy = accuracy_score(val_y, val_prediction)
    print(f'Model accuracy: {accuracy}')
    y_pred_proba = model.predict_proba(val_X)[:, 1]
    auc = roc_auc_score(val_y, y_pred_proba)
    print(f'Model auc score: {auc}')
    return accuracy, auc

# Function to save model
def save_model(model, model_path):
    joblib.dump(model, model_path)

# Usage
file_path = file_path_ # Update with your file path
data = load_data(file_path)
X, y = preprocess_data(data)
train_X, val_X, train_y, val_y = split_data(X, y)
model_xgb = train_model_xgb(train_X, train_y)
accuracy_xgb, auc_xgb = evaluate_model(model_xgb, val_X, val_y)
save_model(model_xgb, 'best_xgb_model.pkl')
```

---

# Procedural Approach – XGBoost with RandomizedSearchCV

## Code

```python
def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    selected_features = [
        'CreditScore', 'Geography', 'Gender', 'Age',
        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary'
    ]
    X = data[selected_features]
    y = data[['Exited']]

    X = pd.get_dummies(X, columns=["Geography", "Gender"])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def split_data(X, y):
    return train_test_split(X, y, random_state=0, train_size=0.8)

def train_model_xgb(train_X, train_y):
    model = xgb.XGBClassifier(random_state=0)
    param_dist = {
        'n_estimators': randint(100, 300),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(3, 7),
        'min_child_weight': randint(1, 5),
        'gamma': uniform(0, 0.3),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'reg_alpha': uniform(0, 0.1),
        'reg_lambda': uniform(1, 1)
    }
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, scoring='roc_auc', cv=3, n_iter=50, random_state=0)
    random_search.fit(train_X, train_y.values.ravel())
    return random_search.best_estimator_

def evaluate_model(model, val_X, val_y):
    val_prediction = model.predict(val_X)
    accuracy = accuracy_score(val_y, val_prediction)
    print(f'Model accuracy: {accuracy}')
    y_pred_proba = model.predict_proba(val_X)[:, 1]
    auc = roc_auc_score(val_y, y_pred_proba)
    print(f'Model auc score: {auc}')
    return accuracy, auc

def save_model(model, model_path):
    joblib.dump(model, model_path)

# Usage
file_path = file_path_
data = load_data(file_path)
X, y = preprocess_data(data)
train_X, val_X, train_y, val_y = split_data(X, y)
model_xgb = train_model_xgb(train_X, train_y)
accuracy_xgb, auc_xgb = evaluate_model(model_xgb, val_X, val_y)
save_model(model_xgb, 'best_xgb_model.pkl')
```

---

# OOP Approach – LightGBM Classifier

## Code

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
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        selected_features = [
            'CreditScore', 'Geography', 'Gender', 'Age',
            'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'EstimatedSalary'
        ]
        self.X = self.data[selected_features]
        self.y = self.data[['Exited']]

        self.X = pd.get_dummies(self.X, columns=["Geography", "Gender"])

    def split_data(self):
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.X, self.y, random_state=0, train_size=0.8
        )

    def train_model(self):
        self.model = lgb.LGBMClassifier(random_state=0)
        self.model.fit(self.train_X, self.train_y)

    def evaluate_model(self):
        val_prediction = self.model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, val_prediction)
        print(f'Model accuracy: {accuracy}')
        y_pred_proba = self.model.predict_proba(self.val_X)[:, 1]
        auc = roc_auc_score(self.val_y, y_pred_proba)
        print(f'Model AUC score: {auc}')
        return accuracy, auc

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

# Usage
churn = ChurnPrediction(file_path_)
churn.load_data()
churn.preprocess_data()
churn.split_data()
churn.train_model()
accuracy, auc = churn.evaluate_model()

# Save the model
churn.save_model('churn_model.pkl')
```

---

# Procedural Approach – LightGBM with GridSearchCV

## Code

```python
def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    selected_features = [
        'CreditScore', 'Geography', 'Gender', 'Age',
        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary'
    ]
    X = data[selected_features]
    y = data[['Exited']]

    X = pd.get_dummies(X, columns=["Geography", "Gender"])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def split_data(X, y):
    return train_test_split(X, y, random_state=0, train_size=0.8)

def train_model_lgb(train_X, train_y):
    model = lgb.LGBMClassifier(random_state=0)
    param_grid = {
                'n_estimators': [100, 200,500,700,800],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50,70],
            }
    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3)
    grid_search.fit(train_X, train_y.values.ravel())
    return grid_search.best_estimator_

def evaluate_model(model, val_X, val_y):
    val_prediction = model.predict(val_X)
    accuracy = accuracy_score(val_y, val_prediction)
    print(f'Model accuracy: {accuracy}')
    y_pred_proba = model.predict_proba(val_X)[:, 1]
    auc = roc_auc_score(val_y, y_pred_proba)
    print(f'Model auc score: {auc}')
    return accuracy, auc

def save_model(model, model_path):
    joblib.dump(model, model_path)

# Usage
file_path = file_path_
data = load_data(file_path)
X, y = preprocess_data(data)
train_X, val_X, train_y, val_y = split_data(X, y)
model_lgb = train_model_lgb(train_X, train_y)
accuracy_lgb, auc_lgb = evaluate_model(model_lgb, val_X, val_y)
save_model(model_lgb, 'best_lgb_model.pkl')
```

---

# Procedural Approach – LightGBM with RandomizedSearchCV

## Code

```python
def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    selected_features = [
        'CreditScore', 'Geography', 'Gender', 'Age',
        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary'
    ]
    X = data[selected_features]
    y = data[['Exited']]

    X = pd.get_dummies(X, columns=["Geography", "Gender"])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def split_data(X, y):
    return train_test_split(X, y, random_state=0, train_size=0.8)

def train_model_lgb(train_X, train_y):
    model = lgb.LGBMClassifier(random_state=0)
    param_dist = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50],
            }
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, scoring='roc_auc', cv=3, n_iter=50, random_state=0)
    random_search.fit(train_X, train_y.values.ravel())
    return random_search.best_estimator_

def evaluate_model(model, val_X, val_y):
    val_prediction = model.predict(val_X)
    accuracy = accuracy_score(val_y, val_prediction)
    print(f'Model accuracy: {accuracy}')
    y_pred_proba = model.predict_proba(val_X)[:, 1]
    auc = roc_auc_score(val_y, y_pred_proba)
    print(f'Model auc score: {auc}')
    return accuracy, auc

def save_model(model, model_path):
    joblib.dump(model, model_path)

# Usage
file_path = file_path_
data = load_data(file_path)
X, y = preprocess_data(data)
train_X, val_X, train_y, val_y = split_data(X, y)
model_lgb = train_model_lgb(train_X, train_y)
accuracy_lgb, auc_lgb = evaluate_model(model_lgb, val_X, val_y)
save_model(model_lgb, 'best_lgb_model.pkl')
```

---

# OOP Approach – Combined XGBoost and LightGBM with GridSearchCV

## Code

```python
class ChurnPrediction:
    def __init__(self, file_path, model_type='xgboost'):
        self.file_path = file_path
        self.model_type = model_type
        self.data = None
        self.X = None
        self.y = None
        self.train_X = None
        self.val_X = None
        self.train_y = None
        self.val_y = None
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        selected_features = [
            'CreditScore', 'Geography', 'Gender', 'Age',
            'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'EstimatedSalary'
        ]
        self.X = self.data[selected_features]
        self.y = self.data[['Exited']]

        self.X = pd.get_dummies(self.X, columns=["Geography", "Gender"])

        # Feature Scaling
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def split_data(self):
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.X, self.y, random_state=0, train_size=0.8
        )

    def train_model(self):
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(random_state=0)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],

            }
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(random_state=0)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70],

            }

        # Using GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(self.model, param_grid, scoring='roc_auc', cv=3)
        grid_search.fit(self.train_X, self.train_y.values.ravel())

        # Setting the best model
        self.model = grid_search.best_estimator_

    def evaluate_model(self):
        val_prediction = self.model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, val_prediction)
        print(f'Model accuracy: {accuracy}')
        y_pred_proba = self.model.predict_proba(self.val_X)[:, 1]
        auc = roc_auc_score(self.val_y, y_pred_proba)
        print(f'Model AUC score: {auc}')
        return accuracy, auc

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

# Ensure the file path is correct
#file_path = file_path_  # Replace this with the actual path to your CSV file
file_path_ = '/content/drive/My Drive/Churn Project/churn.csv'
# Usage for XGBoost
churn_xgb = ChurnPrediction(file_path_, model_type='xgboost')
churn_xgb.load_data()
churn_xgb.preprocess_data()
churn_xgb.split_data()
churn_xgb.train_model()
accuracy_xgb, auc_xgb = churn_xgb.evaluate_model()
churn_xgb.save_model('churn_xgb_model.pkl')

# Usage for LightGBM
churn_lgb = ChurnPrediction(file_path_, model_type='lightgbm')
churn_lgb.load_data()
churn_lgb.preprocess_data()
churn_lgb.split_data()
churn_lgb.train_model()
accuracy_lgb, auc_lgb = churn_lgb.evaluate_model()
churn_lgb.save_model('churn_lgb_model.pkl')
```

---

# OOP Approach – Combined XGBoost and LightGBM with RandomizedSearchCV

## Code

```python
class ChurnPrediction:
    def __init__(self, file_path, model_type='xgboost'):
        self.file_path = file_path
        self.model_type = model_type
        self.data = None
        self.X = None
        self.y = None
        self.train_X = None
        self.val_X = None
        self.train_y = None
        self.val_y = None
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        selected_features = [
            'CreditScore', 'Geography', 'Gender', 'Age',
            'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'EstimatedSalary'
        ]
        self.X = self.data[selected_features]
        self.y = self.data[['Exited']]

        self.X = pd.get_dummies(self.X, columns=["Geography", "Gender"])

        # Feature Scaling
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def split_data(self):
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.X, self.y, random_state=0, train_size=0.8
        )

    def train_model(self):
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(random_state=0)
            param_dist = {
                'n_estimators': stats.randint(100, 300),
                'learning_rate': stats.uniform(0.01, 0.1),
                'max_depth': stats.randint(3, 7),

            }
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(random_state=0)
            param_dist = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 70],

            }


        # Using RandomizedSearchCV for hyperparameter tuning
        random_search = RandomizedSearchCV(self.model, param_dist, n_iter=100, scoring='roc_auc', cv=3)
        random_search.fit(self.train_X, self.train_y.values.ravel())

        # Setting the best model
        self.model = random_search.best_estimator_

    def evaluate_model(self):
        val_prediction = self.model.predict(self.val_X)
        accuracy = accuracy_score(self.val_y, val_prediction)
        print(f'Model accuracy: {accuracy}')
        y_pred_proba = self.model.predict_proba(self.val_X)[:, 1]
        auc = roc_auc_score(self.val_y, y_pred_proba)
        print(f'Model AUC score: {auc}')
        return accuracy, auc

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

# Usage for XGBoost
churn_xgb = ChurnPrediction(file_path_, model_type='xgboost')
churn_xgb.load_data()
churn_xgb.preprocess_data()
churn_xgb.split_data()
churn_xgb.train_model()
accuracy_xgb, auc_xgb = churn_xgb.evaluate_model()
churn_xgb.save_model('churn_xgb_model.pkl')

# Usage for LightGBM
churn_lgb = ChurnPrediction(file_path_, model_type='lightgbm')
churn_lgb.load_data()
churn_lgb.preprocess_data()
churn_lgb.split_data()
churn_lgb.train_model()
accuracy_lgb, auc_lgb = churn_lgb.evaluate_model()
churn_lgb.save_model('churn_lgb_model.pkl')
```

---

# Commands Executed

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score
from sklearn.svm import SVC
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import uniform, randint
import scipy.stats as stats

from google.colab import drive
drive.mount('/content/drive')

file_path_ = '/content/drive/My Drive/Churn Project/churn.csv'
```

---

# Screenshots of Results 

## 1. Library Import and Google Drive Mount

<img width="563" height="213" alt="image" src="https://github.com/user-attachments/assets/fd6346b0-e3e9-44f8-af79-571ab2173630" />



## 2. OOP XGBoost Result

<img width="427" height="395" alt="image" src="https://github.com/user-attachments/assets/4e3c051f-f456-4d6a-864f-6e52705f7e34" />

<img width="406" height="111" alt="image" src="https://github.com/user-attachments/assets/655fe877-7138-48e1-86f6-023e9d6e2717" />



---

## 3. Procedural XGBoost GridSearchCV Result

<img width="427" height="398" alt="image" src="https://github.com/user-attachments/assets/06100b02-fd65-443b-9265-2583a33174b0" />

<img width="413" height="205" alt="image" src="https://github.com/user-attachments/assets/d97ec9ec-37c3-4de2-ba62-dbec6235f46d" />
---

## 4. OOP LightGBM Result

<img width="426" height="392" alt="image" src="https://github.com/user-attachments/assets/7f0f3b27-d356-4577-8189-e4bed1617707" />

<img width="380" height="213" alt="image" src="https://github.com/user-attachments/assets/985f1d81-aafb-4b6e-98f9-1995cd4d62c9" />



---

## 5. Procedural LightGBM GridSearchCV Result

<img width="406" height="392" alt="image" src="https://github.com/user-attachments/assets/42609806-301c-41ff-b406-80b659fbbe40" />

<img width="497" height="170" alt="image" src="https://github.com/user-attachments/assets/14703bb1-553a-4422-8429-225ec6411206" />


---

## 6. Procedural Approach for LightGBM with Automated Hyperparameter Tuning via RandomizedSearchCV


<img width="515" height="344" alt="image" src="https://github.com/user-attachments/assets/e83f0206-a420-4a0c-85bb-46b69339dcab" />

<img width="476" height="79" alt="image" src="https://github.com/user-attachments/assets/a67e78a1-3e3e-4548-b98d-141b0dacc6c1" />

<img width="533" height="178" alt="image" src="https://github.com/user-attachments/assets/0b4e3cb2-f145-43a7-8eff-c76e79f81bae" />



---

## 7. OOP Approach using XGBoost and LightGBM with Automated Hyperparameter Tuning via GridSearchCV

<img width="580" height="389" alt="image" src="https://github.com/user-attachments/assets/ce5d6f2b-10f0-46b1-866e-dfe65fae3d1f" />

<img width="302" height="395" alt="image" src="https://github.com/user-attachments/assets/d18921ed-7f59-4ee7-b481-c15299247516" />

<img width="403" height="391" alt="image" src="https://github.com/user-attachments/assets/45db16ab-5127-49b3-b2f1-bf2787e8c55d" />
<img width="402" height="202" alt="image" src="https://github.com/user-attachments/assets/abffb033-1c4d-47cd-ab84-ecf48f2c2ae0" />

---

## 8. Procedural Approach using XGBoost and LightGBM with Automated Hyperparameter Tuning via RandomizedSearchCV

<img width="430" height="398" alt="image" src="https://github.com/user-attachments/assets/c6f804de-de5d-45c3-85bf-b616840f60c7" />

<img width="407" height="398" alt="image" src="https://github.com/user-attachments/assets/05e93c58-e90e-492b-a538-5ed3b4453147" />

<img width="415" height="298" alt="image" src="https://github.com/user-attachments/assets/cca51f37-71df-4b2f-9e77-4fa737fc3878" />

<img width="344" height="201" alt="image" src="https://github.com/user-attachments/assets/5b1944c0-9928-4a80-a71d-b3c478853770" />


---

# Key Observations

* One-hot encoding was used for `Geography` and `Gender`
* `StandardScaler` was used for feature scaling in the procedural and combined workflows
* XGBoost and LightGBM were both explored as advanced boosting models
* Hyperparameter tuning was performed using both `GridSearchCV` and `RandomizedSearchCV`
* OOP and procedural approaches both support structured machine learning development

---

# Lessons Learned

From this lab, the following skills were gained:

* Loading data from Google Drive in Colab
* Preprocessing churn dataset features
* Applying one-hot encoding and feature scaling
* Training XGBoost and LightGBM models
* Using `GridSearchCV` and `RandomizedSearchCV` for tuning
* Implementing OOP and procedural machine learning workflows
* Saving optimized models for reuse







