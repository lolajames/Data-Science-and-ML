
# Lab Title
## Exploratory Data Analysis (EDA) Using Python

---

# Objective

The objective of this lab is to reproduce classroom exercises demonstrating the fundamentals of **Exploratory Data Analysis (EDA)**.

The lab focuses on:

- Understanding the purpose of EDA
- Inspecting datasets
- Identifying data types and structure
- Handling missing values
- Generating summary statistics
- Exploring relationships within the dataset
- Using visualization techniques to understand patterns

This lab helps build foundational skills required for **data analysis, data cleaning, and preparing datasets for machine learning models**.

---

# Tools Used

The following tools and technologies were used during this lab:

- **Python** – programming language used for the analysis  
- **Pandas** – for data manipulation and analysis  
- **NumPy** – for numerical operations  
- **Matplotlib** – for data visualization  
- **Seaborn** (if used) – for advanced statistical visualization  
- **Jupyter Notebook / Google Colab** – environment used to run the code  
- **GitHub** – for documentation and version control  

---

# Step-by-Step Process

---

# Step 1 – Import Required Libraries

The first step is to import the necessary libraries.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
````

---

# Step 2 – Load the Dataset

The dataset is loaded into a Pandas DataFrame.

```python
df = pd.read_csv('your_dataset.csv')
df.head()
```

This displays the first few rows of the dataset.

---

# Step 3 – Inspect Dataset Structure

```python
df.info()
```

This shows:

* number of rows and columns
* data types
* missing values

---

# Step 4 – Check for Missing Values

```python
df.isnull().sum()
```

This helps identify missing data in each column.

---

# Step 5 – Generate Summary Statistics

```python
df.describe()
```

This provides:

* mean
* median
* standard deviation
* min and max values

---

# Step 6 – Data Cleaning (if applicable)

Handling missing values:

```python
df = df.dropna()
```

or

```python
df = df.fillna(df.mean())
```

---

# Step 7 – Data Visualization

## Histogram

```python
df.hist(figsize=(10,8))
plt.show()
```

---

## Correlation Heatmap

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

## Scatter Plot

```python
plt.scatter(df['column1'], df['column2'])
plt.xlabel('column1')
plt.ylabel('column2')
plt.show()
```

---

# Commands Executed

# Commands Executed

Below are the commands executed during the Exploratory Data Analysis (EDA) lab.

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Display first 5 rows
df.head()

# Display last 5 rows
df.tail()

# Check dataset shape
df.shape

# Check dataset structure and data types
df.info()

# Check for missing values
df.isnull().sum()

# Generate summary statistics
df.describe()

# Check for duplicate values
df.duplicated().sum()

# Remove duplicates (if any)
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# Alternatively (if used)
# df = df.fillna(df.mean())

# View column names
df.columns

# Histogram
df.hist(figsize=(10,8))
plt.show()

# Correlation matrix
corr_matrix = df.corr()

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Scatter plot (example columns)
plt.figure(figsize=(8,6))
plt.scatter(df.iloc[:,0], df.iloc[:,1])
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title('Scatter Plot')
plt.show()

# Box plot
plt.figure(figsize=(8,6))
sns.boxplot(data=df)
plt.show()

# Pairplot (if used)
sns.pairplot(df)
plt.show()
```

---

# Screenshots of Results

Include screenshots showing:

* Dataset preview (`df.head()`)
* Initial Data Inspection
* Summary statistics (`df.describe()`)
* Missing values output
* Bar Plot for Passenger Class (Pclass)
* Box Plot for Age by Passenger Class
* Age Distribution: Histogram
* Survival Rate by Passenger Class
* Correlation of the numerical features: Heatmap
* Scatter plot

<img width="730" height="331" alt="image" src="https://github.com/user-attachments/assets/823a0910-fbd2-4875-9aaf-0228a13991a0" />

<img width="442" height="460" alt="image" src="https://github.com/user-attachments/assets/180f0b7a-583c-4c92-8084-c4c526165336" />

<img width="933" height="372" alt="image" src="https://github.com/user-attachments/assets/c88f1c3e-badc-43ab-8e5c-0f99f1df98de" />

<img width="754" height="438" alt="image" src="https://github.com/user-attachments/assets/d447fc5c-8c8c-44c9-bacf-e6c2419838bc" />

<img width="461" height="215" alt="image" src="https://github.com/user-attachments/assets/0a2cc102-e021-4753-8581-2b5b960c7bf9" />

<img width="462" height="547" alt="image" src="https://github.com/user-attachments/assets/9c71546a-1392-4ef3-9795-6bc6e03d150e" />

<img width="883" height="421" alt="image" src="https://github.com/user-attachments/assets/a61cb62a-0921-4c69-9e52-9a03f3cab278" />

<img width="383" height="357" alt="image" src="https://github.com/user-attachments/assets/3824af32-f7d5-4e91-9ad9-7b0c1d733e5b" />

<img width="435" height="372" alt="image" src="https://github.com/user-attachments/assets/99d0c761-a27a-417d-b82f-b58ab4d4d214" />

<img width="387" height="365" alt="image" src="https://github.com/user-attachments/assets/1b4ab5fa-95de-4506-b0d1-42dcc171719d" />

<img width="365" height="316" alt="image" src="https://github.com/user-attachments/assets/b5d9e57f-b55c-4698-ab24-c86932868366" />

<img width="368" height="368" alt="image" src="https://github.com/user-attachments/assets/d0e2c228-6712-45b7-98e9-d4829683af28" />

<img width="364" height="374" alt="image" src="https://github.com/user-attachments/assets/23fef5b2-758a-4869-af6c-f9057ea61ff0" />

<img width="584" height="415" alt="image" src="https://github.com/user-attachments/assets/a538a2e9-bdf6-4a1b-a448-212ea6c91084" />

<img width="379" height="324" alt="image" src="https://github.com/user-attachments/assets/286effdb-1bef-4f1c-a9ac-1cfe9436b0c6" />


---

# Key Observations

During this lab the following observations were made:

* EDA helps in understanding dataset structure and quality
* Missing values can affect analysis and must be handled
* Summary statistics provide insights into data distribution
* Visualization helps identify patterns and relationships
* Correlation heatmaps reveal relationships between variables

---

# Lessons Learned

From this lab, the following skills were gained:

* Performing Exploratory Data Analysis (EDA)
* Cleaning and preparing datasets
* Using Pandas for data inspection
* Identifying missing values and handling them
* Creating visualizations using Matplotlib and Seaborn
* Interpreting data patterns and relationships







