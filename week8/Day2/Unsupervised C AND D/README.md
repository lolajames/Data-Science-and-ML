
# Lab Title
## Week 8 Day 2 – Unsupervised Learning (C & D)

---

# Objective

The objective of this lab is to reproduce classroom exercises demonstrating advanced **Unsupervised Learning techniques** using clustering and cluster evaluation.

The lab focuses on:

- loading and inspecting the dataset
- cleaning the dataset
- preprocessing numerical and categorical features
- scaling numerical features
- encoding categorical features
- applying **KMeans clustering**
- applying **Agglomerative clustering**
- using **silhouette score** for evaluation
- using **dendrogram visualization**
- determining the optimal number of clusters

This lab builds foundational skills required for **customer segmentation, clustering analysis, and unsupervised machine learning workflows**.

---

# Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook / Google Colab
- GitHub

---

# Step-by-Step Process for Part C

## Step 1 – Connect Drive Path (commented in notebook)

```python
#from google.colab import drive
#drive.mount('/content/drive')
````

```python
#file_path = '/content/drive/My Drive/Retail/Online Retail.xlsx'
```

---

## Step 2 – Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
```

---

## Step 3 – Load the Data

```python
#load the data
customers = pd.read_excel('/content/Online Retail.xlsx')
```

---

## Step 4 – View the First Five Rows

```python
#view the first five rows of the data
customers.head()
```

---

## Step 5 – Check Columns Information

```python
#check columns information
customers.info()
```

---

## Step 6 – Drop Irrelevant Columns

```python
# drop irrelevant columns
customers = customers.drop(columns=['CustomerID','InvoiceDate','InvoiceNo','StockCode','Description'],axis=1)
```

---

## Step 7 – Drop Missing Values

```python
#drop missing values
customers = customers.dropna()
```

---

## Step 8 – Check Columns Information Again

```python
#check columns information
customers.info()
```

---

## Step 9 – Reduce Dataset to 4000 Rows

```python
# reduce dataset to 4000
customers = customers.sample(n=4000,random_state=42)
```

```python
customers.info()
```

---

## Step 10 – Scale Numerical Features

```python
#scaling the numerical features
X = customers[['Quantity','UnitPrice']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Step 11 – Encode Categorical Feature

```python
#encoding the categorical features
label_encoder = LabelEncoder()
countries_encoded = label_encoder.fit_transform(customers['Country'])
```

---

## Step 12 – Check Shape of Encoded Feature

```python
#check the shape of the encoded categorical feature
countries_encoded.shape
```

---

## Step 13 – Check Shape of Scaled Numerical Features

```python
#check the shape of the sclaed numerical features
X_scaled.shape
```

---

## Step 14 – Reshape Encoded Feature to 2D

```python
#reshaping the encoded categorical feature to 2D
countries_encoded = countries_encoded.reshape(-1,1)
```

---

## Step 15 – View the Shape Again

```python
#view the shape of the encoded categorical feature
countries_encoded.shape
```

---

## Step 16 – Concatenate Numerical and Encoded Features

```python
#Concatenating the scaled numerical features and the encoded categorical feature
X_scaled_with_countries = np.concatenate((X_scaled,countries_encoded),axis=1)
```

---

## Step 17 – Calculate Silhouette Scores for KMeans

```python
#Silhouette Score Calculation
silhouette_scores = []
cluster_range = range(2,16)
for n in cluster_range:
  kmeans = KMeans(n_clusters=n,random_state=42,n_init='auto')
  kmeans_labels = kmeans.fit_predict(X_scaled_with_countries)
  silhouette_scores.append(silhouette_score(X_scaled_with_countries,kmeans_labels))
```

---

## Step 18 – Visualize Silhouette Scores

```python
plt.figure(figsize=(10,4))
plt.plot(cluster_range,silhouette_scores,marker='o')
plt.title('Visualization of the silhouette_scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(np.arange(2,16,step=1))
plt.grid(True)
plt.show()
```

---

## Step 19 – Find Optimal Number of Clusters

```python
#find the optimal number of clusters
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print("optimal number of clusters",optimal_clusters)
```

---

## Step 20 – Fit Final KMeans Model and Evaluate

```python
kmeans = KMeans(n_clusters = optimal_clusters,random_state=42,n_init='auto')
kmeans_labels_ = kmeans.fit_predict(X_scaled_with_countries)
score = silhouette_score(X_scaled_with_countries,kmeans_labels_)
print('Silhouette_score : ',score)
```

---

# Step-by-Step Process for Part D

## Step 1 – Connect Drive Path (commented in notebook)

```python
#from google.colab import drive
#drive.mount('/content/drive')
```

---

## Step 2 – Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')
```

```python
import warnings
warnings.filterwarnings('ignore')
```

---

## Step 3 – Load Data

```python
# Load data
customers = pd.read_excel('/content/Online Retail.xlsx')
```

---

## Step 4 – View First Rows

```python
customers.head()
```

---

## Step 5 – Inspect Data

```python
customers.info()
```

---

## Step 6 – Clean Data

```python
# Data cleaning
customers = customers.drop(columns=['StockCode', 'Description', 'InvoiceDate', 'CustomerID', 'InvoiceNo'], axis=1)
```

```python
customers = customers.sample(n=4000, random_state=42)
```

```python
customers = customers.dropna()
```

---

## Step 7 – Preprocess Numerical Features

```python
# Preprocessing
X = customers[['Quantity', 'UnitPrice']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Step 8 – Encode Categorical Variable

```python
# Encoding Categorical variables
countries = customers['Country']
label_encoder = LabelEncoder()
countries_encoded = label_encoder.fit_transform(countries).reshape(-1, 1)
```

---

## Step 9 – Combine Numerical and Encoded Features

```python
X_scaled_with_countries = np.concatenate((X_scaled, countries_encoded), axis=1)
```

---

## Step 10 – Perform Hierarchical Clustering

```python
# Perform hierarchical clustering (Agglomerative Clustering)
linked = linkage(X_scaled_with_countries, method='ward')
```

---

## Step 11 – Plot Dendrogram

```python
# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', truncate_mode='level', p=5, no_labels=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
```

---

## Step 12 – Fit Agglomerative Clustering with 5 Clusters

```python
agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
clusters = agg_clustering.fit_predict(X_scaled_with_countries)
```

---

## Step 13 – Compute Silhouette Score

```python
score = silhouette_score(X_scaled_with_countries, clusters)
```

```python
print("Agglomerative Clustering Silhouette score:", score)
```

---

## Step 14 – Reload and Reprocess Data

```python
# Load data
customers = pd.read_excel('/content/Online Retail.xlsx')
```

```python
# Data cleaning
customers = customers.drop(columns=['StockCode', 'Description', 'InvoiceDate', 'CustomerID', 'InvoiceNo'], axis=1)
customers = customers.sample(n=4000, random_state=42)
customers = customers.dropna()
```

```python
# Preprocessing
X = customers[['Quantity', 'UnitPrice']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

```python
countries = customers['Country']
label_encoder = LabelEncoder()
countries_encoded = label_encoder.fit_transform(countries).reshape(-1, 1)
```

```python
X_scaled_with_countries = np.concatenate((X_scaled, countries_encoded), axis=1)
```

---

## Step 15 – Perform Hierarchical Clustering Again

```python
# Perform hierarchical clustering (Agglomerative Clustering)
linked = linkage(X_scaled_with_countries, method='ward')
```

---

## Step 16 – Plot Dendrogram Again

```python
# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', truncate_mode='level', p=5, no_labels=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
```

---

## Step 17 – Calculate Silhouette Scores Across Cluster Counts

```python
# Silhouette Score Calculation
silhouette_scores = []

for nr_c in range(2, 16):  # Start from 2 clusters, as silhouette score requires at least 2 clusters
    agglomerative = AgglomerativeClustering(n_clusters=nr_c)
    agglomerative_labels = agglomerative.fit_predict(X_scaled_with_countries)
    silhouette_scores.append(silhouette_score(X_scaled_with_countries, agglomerative_labels))
```

---

## Step 18 – Prepare Cluster Range

```python
# Elbow method
cluster_range = range(2, 16)
silhouette_scores_subset = silhouette_scores[:14]  # Take subset as silhouette scores are calculated from 2 to 15 clusters
```

---

## Step 19 – Plot Silhouette Scores for Agglomerative Clustering

```python
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores_subset, marker='o')
plt.title('Elbow method for Agglomerative Clustering using Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.xticks(np.arange(2, 16, step=1))  # Adjust ticks for better visualization
plt.grid(True)
plt.show()
```

---

## Step 20 – Find Optimal Number of Clusters

```python
# Find the optimal number of clusters
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 as we started from 2 clusters
print("Optimal number of clusters:", optimal_clusters)
```

---

## Step 21 – Fit Final Agglomerative Model

```python
# Agglomerative Clustering with optimal number of clusters
agglomerative = AgglomerativeClustering(n_clusters=optimal_clusters)
agglomerative_labels = agglomerative.fit_predict(X_scaled_with_countries)
```

---

## Step 22 – Final Silhouette Score

```python
# Silhouette score for Agglomerative Clustering
agglomerative_score = silhouette_score(X_scaled_with_countries, agglomerative_labels)
print("Agglomerative Clustering Silhouette score:", agglomerative_score)
```

---

# Commands Executed Part C

```python
#from google.colab import drive
#drive.mount('/content/drive')

#file_path = '/content/drive/My Drive/Retail/Online Retail.xlsx'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

#load the data
customers = pd.read_excel('/content/Online Retail.xlsx')

#view the first five rows of the data
customers.head()

#check columns information
customers.info()

# drop irrelevant columns
customers = customers.drop(columns=['CustomerID','InvoiceDate','InvoiceNo','StockCode','Description'],axis=1)

#drop missing values
customers = customers.dropna()

#check columns information
customers.info()

# reduce dataset to 4000
customers = customers.sample(n=4000,random_state=42)

customers.info()

#scaling the numerical features
X = customers[['Quantity','UnitPrice']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#encoding the categorical features
label_encoder = LabelEncoder()
countries_encoded = label_encoder.fit_transform(customers['Country'])

#check the shape of the encoded categorical feature
countries_encoded.shape

#check the shape of the sclaed numerical features
X_scaled.shape

#reshaping the encoded categorical feature to 2D
countries_encoded = countries_encoded.reshape(-1,1)

#view the shape of the encoded categorical feature
countries_encoded.shape

#Concatenating the scaled numerical features and the encoded categorical feature
X_scaled_with_countries = np.concatenate((X_scaled,countries_encoded),axis=1)

#Silhouette Score Calculation
silhouette_scores = []
cluster_range = range(2,16)
for n in cluster_range:
  kmeans = KMeans(n_clusters=n,random_state=42,n_init='auto')
  kmeans_labels = kmeans.fit_predict(X_scaled_with_countries)
  silhouette_scores.append(silhouette_score(X_scaled_with_countries,kmeans_labels))

plt.figure(figsize=(10,4))
plt.plot(cluster_range,silhouette_scores,marker='o')
plt.title('Visualization of the silhouette_scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(np.arange(2,16,step=1))
plt.grid(True)
plt.show()

#find the optimal number of clusters
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print("optimal number of clusters",optimal_clusters)

kmeans = KMeans(n_clusters = optimal_clusters,random_state=42,n_init='auto')
kmeans_labels_ = kmeans.fit_predict(X_scaled_with_countries)
score = silhouette_score(X_scaled_with_countries,kmeans_labels_)
print('Silhouette_score : ',score)
```

---

# Commands Executed Part D

```python
#from google.colab import drive
#drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')

# Load data
customers = pd.read_excel('/content/Online Retail.xlsx')

customers.head()

customers.info()

# Data cleaning
customers = customers.drop(columns=['StockCode', 'Description', 'InvoiceDate', 'CustomerID', 'InvoiceNo'], axis=1)

customers = customers.sample(n=4000, random_state=42)

customers = customers.dropna()

# Preprocessing
X = customers[['Quantity', 'UnitPrice']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encoding Categorical variables
countries = customers['Country']
label_encoder = LabelEncoder()
countries_encoded = label_encoder.fit_transform(countries).reshape(-1, 1)

X_scaled_with_countries = np.concatenate((X_scaled, countries_encoded), axis=1)

# Perform hierarchical clustering (Agglomerative Clustering)
linked = linkage(X_scaled_with_countries, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', truncate_mode='level', p=5, no_labels=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
clusters = agg_clustering.fit_predict(X_scaled_with_countries)

score = silhouette_score(X_scaled_with_countries, clusters)

print("Agglomerative Clustering Silhouette score:", score)

# Load data
customers = pd.read_excel('/content/Online Retail.xlsx')

# Data cleaning
customers = customers.drop(columns=['StockCode', 'Description', 'InvoiceDate', 'CustomerID', 'InvoiceNo'], axis=1)
customers = customers.sample(n=4000, random_state=42)
customers = customers.dropna()

# Preprocessing
X = customers[['Quantity', 'UnitPrice']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

countries = customers['Country']
label_encoder = LabelEncoder()
countries_encoded = label_encoder.fit_transform(countries).reshape(-1, 1)

X_scaled_with_countries = np.concatenate((X_scaled, countries_encoded), axis=1)

# Perform hierarchical clustering (Agglomerative Clustering)
linked = linkage(X_scaled_with_countries, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', truncate_mode='level', p=5, no_labels=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Silhouette Score Calculation
silhouette_scores = []

for nr_c in range(2, 16):  # Start from 2 clusters, as silhouette score requires at least 2 clusters
    agglomerative = AgglomerativeClustering(n_clusters=nr_c)
    agglomerative_labels = agglomerative.fit_predict(X_scaled_with_countries)
    silhouette_scores.append(silhouette_score(X_scaled_with_countries, agglomerative_labels))

# Elbow method
cluster_range = range(2, 16)
silhouette_scores_subset = silhouette_scores[:14]  # Take subset as silhouette scores are calculated from 2 to 15 clusters

plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores_subset, marker='o')
plt.title('Elbow method for Agglomerative Clustering using Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.xticks(np.arange(2, 16, step=1))  # Adjust ticks for better visualization
plt.grid(True)
plt.show()

# Find the optimal number of clusters
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 as we started from 2 clusters
print("Optimal number of clusters:", optimal_clusters)

# Agglomerative Clustering with optimal number of clusters
agglomerative = AgglomerativeClustering(n_clusters=optimal_clusters)
agglomerative_labels = agglomerative.fit_predict(X_scaled_with_countries)

# Silhouette score for Agglomerative Clustering
agglomerative_score = silhouette_score(X_scaled_with_countries, agglomerative_labels)
print("Agglomerative Clustering Silhouette score:", agglomerative_score)
```

---

# Screenshots of Results Part C

---
# customers.head
<img width="469" height="191" alt="image" src="https://github.com/user-attachments/assets/c53681e0-22d4-4988-a467-816093b79a05" />

# first customers.info()
<img width="536" height="178" alt="image" src="https://github.com/user-attachments/assets/22cb4fa4-1db4-4ab3-bfaa-14d80a376741" />

# second `customers.info()` after cleaning
<img width="386" height="119" alt="image" src="https://github.com/user-attachments/assets/3d68299b-9b60-430c-b4be-2f53a871c99a" />

# `customers.info()` after sampling
<img width="272" height="118" alt="image" src="https://github.com/user-attachments/assets/c89085ba-85ad-4a5d-bdca-5100ed757e64" />

# `countries_encoded.shape`
<img width="404" height="121" alt="image" src="https://github.com/user-attachments/assets/1af92ab9-01a1-4249-ae57-4a9a4b02c972" />

# `X_scaled.shape`
<img width="307" height="79" alt="image" src="https://github.com/user-attachments/assets/8e7f2616-c085-4c17-b1d5-d9f0356817dc" />

# reshaped `countries_encoded.shape`
<img width="323" height="91" alt="image" src="https://github.com/user-attachments/assets/bcddb6a9-33e8-436f-9395-8a5475a6ba1c" />


# silhouette score plot

<img width="422" height="289" alt="image" src="https://github.com/user-attachments/assets/8377323a-ffa9-4b9b-b091-8b845577d2f3" />

# optimal number of clusters output and final KMeans silhouette score output
<img width="441" height="136" alt="image" src="https://github.com/user-attachments/assets/8ec0a023-f074-4e74-a5b5-a44ae94f2631" />

---

# Screenshots of Results Part D


# `customers.head()`
<img width="524" height="278" alt="image" src="https://github.com/user-attachments/assets/f0418500-b008-4983-a01f-fba82ec9fdc3" />

# `customers.info()`
<img width="269" height="123" alt="image" src="https://github.com/user-attachments/assets/c3c6f0e5-a183-47a5-9972-8721628ebd64" />

# Data cleaning and Preprocessing and Encoding Categorical variables and Perform hierarchical clustering (Agglomerative Clustering)
<img width="496" height="208" alt="image" src="https://github.com/user-attachments/assets/a96605b9-ec1c-4b75-a1da-5d92d2fbbb45" />


# first dendrogram plot
<img width="473" height="302" alt="image" src="https://github.com/user-attachments/assets/c13d84e8-9761-44f2-8022-a2232313c9e0" />

# first agglomerative silhouette score output
<img width="457" height="91" alt="image" src="https://github.com/user-attachments/assets/001c2b10-c8cb-44de-9f42-0bcc3db0a330" />

# Using the Elbow Method
<img width="419" height="209" alt="image" src="https://github.com/user-attachments/assets/6bec9d2e-2b44-46c2-aeae-391497e13e09" />

# second dendrogram plot
<img width="460" height="295" alt="image" src="https://github.com/user-attachments/assets/e9bf9991-8171-46a0-aac6-ceea98554f9e" />

# elbow/silhouette plot
<img width="508" height="395" alt="image" src="https://github.com/user-attachments/assets/52cd9412-7ed0-40eb-822a-63df4f9805b0" />

# Find the optimal number of clusters &  Agglomerative Clustering with optimal number of clusters &  Silhouette score for Agglomerative Clustering
<img width="446" height="163" alt="image" src="https://github.com/user-attachments/assets/866cbdc1-fa4e-4420-9fb8-957fe63607c0" />


---

# Key Observations

* Label encoding was used on the `Country` column.
* StandardScaler was used on `Quantity` and `UnitPrice`.
* Part C used **KMeans** clustering and silhouette score to determine the best number of clusters.
* Part D used **Agglomerative Clustering** with dendrogram visualization and silhouette score to determine the best number of clusters.

---

# Lessons Learned

From this lab, the following skills were gained:

* cleaning and sampling a retail dataset
* scaling numerical variables
* encoding categorical variables
* combining numerical and categorical features for clustering
* evaluating KMeans using silhouette score
* performing Agglomerative clustering
* visualizing hierarchical clustering with dendrograms
* selecting optimal cluster counts using silhouette score


