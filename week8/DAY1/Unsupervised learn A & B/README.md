

# Lab Title
## Week 8 Day 1 – Unsupervised Learning (A) and (B)

---

# Objective

The objective of this lab is to reproduce classroom exercises demonstrating the fundamentals of **Unsupervised Learning** using clustering, dimensionality reduction, and customer segmentation techniques.

The lab focuses on:

- understanding unsupervised learning concepts
- loading and inspecting datasets
- preprocessing data
- scaling numerical features
- encoding categorical variables
- applying **PCA (Principal Component Analysis)**
- applying **K-Means Clustering**
- evaluating clustering performance using:
  - inertia
  - silhouette score
  - Davies-Bouldin index
- visualizing clustering results

This lab helps build foundational skills required for **pattern discovery, customer segmentation, clustering analysis, and dimensionality reduction in Machine Learning**.

---

# Tools Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook / Google Colab**
- **GitHub**

---

# Step-by-Step Process

---

# Part A – Unsupervised Learning on the Digits Dataset

## Step 1 – Import Required Libraries

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
````

---

## Step 2 – Load the Digits Dataset

```python
# Load the Digits dataset
digits = load_digits()
data = digits.data
target = digits.target
```

---

## Step 3 – Create a DataFrame

```python
# Create a DataFrame for easy manipulation
df = pd.DataFrame(data)
df['target'] = target
```

---

## Step 4 – Inspect the Dataset

```python
# Display the first few rows of the DataFrame
df.head()
```

```python
df.info()
```

```python
df['target'].value_counts()
```

```python
df['target'].unique()
```

---

## Step 5 – Scale the Data

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)
```

---

## Step 6 – Apply PCA

```python
# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Create a DataFrame for the PCA-transformed data
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['target'] = target
pca_df.head()
```

---

## Step 7 – Apply K-Means Clustering

```python
# Apply K-Means Clustering
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(pca_data)

# Add cluster labels to the PCA DataFrame
pca_df['cluster'] = kmeans.labels_
pca_df['cluster']
```

---

## Step 8 – Visualize Clusters

```python
# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis', s=50)
plt.title('K-Means Clustering on Digits Dataset (PCA-reduced)')
plt.show()
```

---

## Step 9 – Evaluate Clustering Output

```python
# Confusion Matrix
conf_matrix = confusion_matrix(target, kmeans.labels_)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(target, kmeans.labels_))
```

---

# Part B – Mall Customer Segmentation Using K-Means

## Step 1 – Drive Setup

```python
#from google.colab import drive
#drive.mount('/content/drive')
```

```python
#file_path = '/content/drive/My Drive/Mall Dataset/Mall_Customers.csv'
```

---

## Step 2 – Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
```

---

## Step 3 – Load the Dataset

```python
# Load your data
data = pd.read_csv('/content/Mall Customers.csv')
```

---

## Step 4 – Inspect the Dataset

```python
# Inspect the data
data.head()
```

```python
data.info()
```

---

## Step 5 – Identify Categorical and Numerical Columns

```python
# Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=['float', 'int']).columns.tolist()
```

---

## Step 6 – One-Hot Encode Categorical Data

```python
# One-hot encode the categorical columns
ohe = OneHotEncoder()
encoded_categorical_data = ohe.fit_transform(data[categorical_cols]).toarray()
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=ohe.get_feature_names_out(categorical_cols))
print(encoded_categorical_df.head())
```

---

## Step 7 – Scale Numerical Data

```python
# Standard scale the numerical columns
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(data[numerical_cols])
scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_cols)
```

---

## Step 8 – Combine Processed Data

```python
# Combine the processed numerical and categorical data
processed_data = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)
```

```python
processed_data.head()
```

---

## Step 9 – Apply K-Means and Evaluate with Elbow Method, Silhouette Score, and Davies-Bouldin Score

```python
# Apply the KMeans algorithm and evaluate using the elbow method and silhouette score
inertia = []
silhouette_scores = []
davies_bouldin_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(processed_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(processed_data, cluster_labels))
    davies_bouldin_scores.append(davies_bouldin_score(processed_data, cluster_labels))
```

---

## Step 10 – Plot Elbow Method and Silhouette Score

```python
# Plot the elbow method
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k')

plt.tight_layout()
plt.show()
```

---

## Step 11 – Determine Optimal Number of Clusters

```python
# Determine the best number of clusters using the silhouette score
optimal_k = K[np.argmax(silhouette_scores)]
print(f"The optimal number of clusters based on silhouette score is: {optimal_k}")
```

---

## Step 12 – Fit Final K-Means Model

```python
# Fit the final KMeans model with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_optimal.fit(processed_data)

# Add cluster labels to the original dataset
data['Cluster'] = kmeans_optimal.labels_

# Display the final inertia and silhouette score
print(f"Final Inertia: {kmeans_optimal.inertia_}")
print(f"Final Silhouette Score: {silhouette_score(processed_data, kmeans_optimal.labels_)}")
print(f"Final Davies-Bouldin Index: {davies_bouldin_score(processed_data, kmeans_optimal.labels_)}")
```

---

## Step 13 – Apply PCA

```python
# Use PCA for dimensionality reduction

# Apply PCA and keep the top 2 principal components
pca = PCA(n_components=2)
pca_data = pca.fit_transform(processed_data)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])

# Add the cluster labels
pca_df['Cluster'] = kmeans_optimal.labels_
```

---

## Step 14 – K-Means Evaluation on PCA Data

```python
# Plot the elbow method for PCA data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k (PCA Data)')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k (PCA Data)')

plt.tight_layout()
plt.show()
```

---

## Step 15 – Determine Optimal Clusters for PCA Data

```python
# Determine the best number of clusters using the silhouette score for PCA data
optimal_k_pca = K[np.argmax(silhouette_scores_pca)]
print(f"The optimal number of clusters based on silhouette score for PCA data is: {optimal_k_pca}")
```

---

## Step 16 – Final K-Means on PCA Data

```python
# Fit the final KMeans model with the optimal number of clusters on PCA data
kmeans_optimal_pca = KMeans(n_clusters=optimal_k_pca, random_state=42)
kmeans_optimal_pca.fit(pca_data)

# Add cluster labels to the PCA DataFrame
pca_df['Cluster_PCA'] = kmeans_optimal_pca.labels_

# Display the final inertia and silhouette score for PCA
print(f"Final Inertia (PCA): {kmeans_optimal_pca.inertia_}")
print(f"Final Silhouette Score (PCA): {silhouette_score(pca_data, kmeans_optimal_pca.labels_)}")
print(f"Final Davies-Bouldin Index (PCA): {davies_bouldin_score(pca_data, kmeans_optimal_pca.labels_)}")
```

---

# Commands Executed

## Unsupervised Learning (A)

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Digits dataset
digits = load_digits()
data = digits.data
target = digits.target

# Create a DataFrame for easy manipulation
df = pd.DataFrame(data)
df['target'] = target

# Display the first few rows of the DataFrame
df.head()

df.info()
df['target'].value_counts()
df['target'].unique()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Create a DataFrame for the PCA-transformed data
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['target'] = target
pca_df.head()

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(pca_data)

# Add cluster labels to the PCA DataFrame
pca_df['cluster'] = kmeans.labels_
pca_df['cluster']

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis', s=50)
plt.title('K-Means Clustering on Digits Dataset (PCA-reduced)')
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(target, kmeans.labels_)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(target, kmeans.labels_))
```

## Unsupervised Learning (B)

```python
#from google.colab import drive
#drive.mount('/content/drive')

#file_path = '/content/drive/My Drive/Mall Dataset/Mall_Customers.csv'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load your data
data = pd.read_csv('/content/Mall Customers.csv')

# Inspect the data
data.head()
data.info()

# Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=['float', 'int']).columns.tolist()

# One-hot encode the categorical columns
ohe = OneHotEncoder()
encoded_categorical_data = ohe.fit_transform(data[categorical_cols]).toarray()
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=ohe.get_feature_names_out(categorical_cols))
print(encoded_categorical_df.head())

# Standard scale the numerical columns
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(data[numerical_cols])
scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_cols)

# Combine the processed numerical and categorical data
processed_data = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)
processed_data.head()

# Apply the KMeans algorithm and evaluate using the elbow method and silhouette score
inertia = []
silhouette_scores = []
davies_bouldin_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(processed_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(processed_data, cluster_labels))
    davies_bouldin_scores.append(davies_bouldin_score(processed_data, cluster_labels))

# Plot the elbow method
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k')

plt.tight_layout()
plt.show()

# Determine the best number of clusters using the silhouette score
optimal_k = K[np.argmax(silhouette_scores)]
print(f"The optimal number of clusters based on silhouette score is: {optimal_k}")

# Fit the final KMeans model with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_optimal.fit(processed_data)

# Add cluster labels to the original dataset
data['Cluster'] = kmeans_optimal.labels_

# Display the final inertia and silhouette score
print(f"Final Inertia: {kmeans_optimal.inertia_}")
print(f"Final Silhouette Score: {silhouette_score(processed_data, kmeans_optimal.labels_)}")
print(f"Final Davies-Bouldin Index: {davies_bouldin_score(processed_data, kmeans_optimal.labels_)}")

# Use PCA for dimensionality reduction

# Apply PCA and keep the top 2 principal components
pca = PCA(n_components=2)
pca_data = pca.fit_transform(processed_data)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])

# Add the cluster labels
pca_df['Cluster'] = kmeans_optimal.labels_

# Plot the elbow method for PCA data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k (PCA Data)')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k (PCA Data)')

plt.tight_layout()
plt.show()

# Determine the best number of clusters using the silhouette score for PCA data
optimal_k_pca = K[np.argmax(silhouette_scores_pca)]
print(f"The optimal number of clusters based on silhouette score for PCA data is: {optimal_k_pca}")

# Fit the final KMeans model with the optimal number of clusters on PCA data
kmeans_optimal_pca = KMeans(n_clusters=optimal_k_pca, random_state=42)
kmeans_optimal_pca.fit(pca_data)

# Add cluster labels to the PCA DataFrame
pca_df['Cluster_PCA'] = kmeans_optimal_pca.labels_

# Display the final inertia and silhouette score for PCA
print(f"Final Inertia (PCA): {kmeans_optimal_pca.inertia_}")
print(f"Final Silhouette Score (PCA): {silhouette_score(pca_data, kmeans_optimal_pca.labels_)}")
print(f"Final Davies-Bouldin Index (PCA): {davies_bouldin_score(pca_data, kmeans_optimal_pca.labels_)}")
```

---

# Screenshots of Results

## Unsupervised Learning (A)

### 1. Dataset Preview

<img width="382" height="357" alt="image" src="https://github.com/user-attachments/assets/bd3a8e65-913f-4082-a068-f19028be3665" />


---

### 2. Dataset Information
<img width="341" height="388" alt="image" src="https://github.com/user-attachments/assets/3655b6d5-1698-4068-998b-081faac0d5f5" />

<img width="268" height="129" alt="image" src="https://github.com/user-attachments/assets/909d8233-e458-4c52-bee1-4c75d629740a" />

---

### 3. Target Distribution

<img width="283" height="184" alt="image" src="https://github.com/user-attachments/assets/4876a8ed-1a2b-4112-b93b-a14806cca713" />

---

### 4. Scaled Data Output

<img width="400" height="130" alt="image" src="https://github.com/user-attachments/assets/bde2d60e-4c6a-4dd0-b244-fb58de417c63" />


---

### 5. PCA DataFrame Output

<img width="322" height="224" alt="image" src="https://github.com/user-attachments/assets/a42bf255-2c46-4293-8ec4-3dc82bd2c569" />


---

### 6. Cluster Labels Output
<img width="389" height="233" alt="image" src="https://github.com/user-attachments/assets/1b053255-d541-404f-b6ad-037fb2d267fe" />


---

### 7. K-Means Cluster Visualization

<img width="449" height="272" alt="image" src="https://github.com/user-attachments/assets/27413051-fb06-4497-9342-9749097dd063" />

---

### 8. Confusion Matrix


<img width="352" height="283" alt="image" src="https://github.com/user-attachments/assets/95375f9c-fbfe-4288-9ea7-ed131292f574" />


---


## Unsupervised Learning (B)

### 10. Mall Dataset Preview

<img width="425" height="257" alt="image" src="https://github.com/user-attachments/assets/eb6e5774-233f-469a-977d-bba6c7c8a32a" />

---

### 11. Mall Dataset Info

<img width="287" height="97" alt="image" src="https://github.com/user-attachments/assets/febf7245-c3ed-4c8c-a3c6-665262bda510" />

---

### 12. Encoded Categorical Data

<img width="424" height="310" alt="image" src="https://github.com/user-attachments/assets/4d310fc2-4b8f-4524-be8f-4d3c0968de91" />


### 13. Processed Data Preview

<img width="476" height="104" alt="image" src="https://github.com/user-attachments/assets/2491dd88-6c8e-4f5a-ba1a-dcf1d8ac4359" />


---

### 14. Elbow Method and Silhouette Score Plot

<img width="467" height="280" alt="image" src="https://github.com/user-attachments/assets/5467af05-d582-4742-a816-60b067e80d98" />

<img width="584" height="208" alt="image" src="https://github.com/user-attachments/assets/38145a00-7323-4ea1-9eb7-6df26c194e7e" />

### 15. Optimal Number of Clusters

<img width="415" height="234" alt="image" src="https://github.com/user-attachments/assets/b4414e97-5079-4b76-9ec4-fe8cc54048ea" />


---

### 16. Final Model Evaluation
<img width="424" height="167" alt="image" src="https://github.com/user-attachments/assets/fa8cb117-1c8e-4d19-b0a4-fa6369e519ea" />

---

### 17. PCA Optimal Clusters

<img width="473" height="293" alt="image" src="https://github.com/user-attachments/assets/7baefdb9-aa5a-4c80-8f65-73cd492690c3" />

<img width="547" height="200" alt="image" src="https://github.com/user-attachments/assets/cb107338-0847-4014-b248-bb64f27abfe1" />

<img width="466" height="70" alt="image" src="https://github.com/user-attachments/assets/f938e854-1f48-4f6f-9a7a-726ed405a9a6" />

---

### 18. Final PCA Model Evaluation

<img width="437" height="163" alt="image" src="https://github.com/user-attachments/assets/3904c96d-b16b-4aec-81e4-f64776670dd3" />


---

# Key Observations

* Unsupervised learning works with unlabeled data.
* PCA helps reduce dimensionality while preserving key patterns.
* K-Means can be used to discover hidden groups in a dataset.
* StandardScaler improves clustering performance by normalizing features.
* The elbow method and silhouette score help identify the best number of clusters.
* Davies-Bouldin index provides an additional clustering quality measure.
* Mall customer segmentation can reveal meaningful customer groups for business decisions.

---

# Lessons Learned

From this lab, the following skills were gained:

* loading and inspecting datasets
* scaling numerical data
* encoding categorical variables
* reducing dimensions using PCA
* applying K-Means clustering
* evaluating clustering performance
* visualizing clusters
* comparing clustering performance before and after PCA








