# Lab Title
## Introduction to NumPy Arrays and Pandas Data Structures

---

# Objective

The objective of this lab is to reproduce classroom exercises demonstrating the fundamentals of **NumPy arrays** and **Pandas data structures**.

The lab focuses on the following tasks:

- Creating NumPy arrays  
- Exploring array properties  
- Performing array arithmetic operations  
- Creating Pandas Series  
- Creating Pandas DataFrames  
- Accessing data within Series and DataFrames  
- Combining DataFrames using concatenation  

This lab helps build foundational skills required for **data manipulation and analysis in Data Science and Machine Learning**.

---

# Tools Used

The following tools and technologies were used during this lab:

- **Python** – programming language used for the exercises  
- **NumPy** – library used for numerical computing and array operations  
- **Pandas** – library used for data manipulation and analysis  
- **Jupyter Notebook / Google Colab** – environment used to write and run the code  
- **GitHub** – used for documentation and version control  

---

# Step-by-Step Process

---

# Part 1 – NumPy Operations

---

## Step 1 – Import NumPy

The first step is to import the **NumPy** library. NumPy is widely used in data science for handling numerical computations and working efficiently with arrays.

```python
import numpy as np
```

NumPy is used for efficient numerical computations and array operations.

# Step 2 – Create a 1D NumPy Array

A one-dimensional array was created from a Python list.
```python
arr1d = np.array([1,2,3,4,5])
arr1d
```
This produces a simple numeric Step 3 – Create a 2D NumPy Array

# A two-dimensional array was created from a nested list.
```python 
arr2d = np.array([[1,2,3],[4,5,6]])
arr2dA 
```
2D array represents rows and columns similar to a table.

# Step 4 – Create an Array of Zeros
NumPy can generate arrays filled with zeros.
```python
zeros_arr = np.zeros((3,4))
print(zeros_arr)
```
This creates a 3×4 matrix filled with zeros.

# Step 5 – Check Array Shape

The shape shows the dimensions of the array.
```python
shape = arr2d.shape
shape
Output:(2,3)
```

Meaning:
	•	2 rows
	•	3 columns
    
# Step 6 – Check Array Size

The size gives the total number of elements in the array.
```python
size = arr2d.size
size
```
# Step 7 – Check Data Type

NumPy arrays store elements of the same type.
```python 
data_type = arr2d.dtype
data_type
```

# Step 8 – Perform Arithmetic Operations

**Addition**
```python 
addition = 2 + arr2d
print(addition)```


**Subtraction**
```python 
substraction = arr2d - 3
print(substraction)
```
**Multiplication**
```python 
Multiplication = 4 * arr2d
print(Multiplication)
```
---
# Pandas Operations

# Step 9 – Install Pandas
```python 
pip install pandas
```
This installs the Pandas library.

## Step 10 – Import Pandas
```python 
import pandas as pd
```

## Step 11 – Create a Pandas Series
A Pandas Series is a one-dimensional labeled array.
```python s = pd.Series([1,2,3,'angel',4,5])
print(s)```

## Step 12 – Create a DataFrame

A DataFrame was created from a dictionary.
```python df = {
'name':["David","sarah","moses"],
'age':[59,20,33],
'location':['boston','adenta','columbus']
}
data = pd.DataFrame(df)
print(data)
```

## Step 13 – Create Another DataFrame
```python 
data = 
{'name':['john','alice','Bod'],
'age':[30,25,35],
'city':['newyork','paris','london']
}

df = pd.DataFrame(data)
print(df)```
## Step 14 – Access Elements in a Series
```python
print(s[0])```
This retrieves the first element of the Series.

## Step 15 – Access DataFrame Columns
```python 
df[['name']] 
```

This returns the name column from the DataFrame.

## Step 16 – Create Multiple DataFramesDataFrames

```python
df_1 = pd.DataFrame({'a':[4,2,6],'b':['m','n','o']})
df_2 = pd.DataFrame({'a':[45,8,9],'b':['x','y','z']})
```


⸻

## Step 17 – Vertical Concatenation

```python
vertical_com = pd.concat([df_1,df_2], 
axis=0, keys=['df_1','df_2'])
vertical_com
```

This combines DataFrames row-wise.

⸻

## Step 18 – Horizontal Concatenation

```python
horizontal_com = pd.concat([df_1,df_2], axis=1, keys=['df_1','df_2'])
horizontal_com
```

This combines DataFrames column-wise.

⸻

## Step 19 – Access Values Using loc

```python 
vertical_com.loc['df_1',0]

```


⸻

## Step 20 – Access Specific Element

```python
vertical_com.loc["df_1",0].iloc[0]
```


⸻

## Commands Executed

## All commands used in the lab:

``` python 
import numpy as np
import pandas as pd

arr1d=np.array([1,2,3,4,5])
arr2d=np.array([[1,2,3],[4,5,6]])

zeros_arr = np.zeros((3,4))

shape=arr2d.shape
size=arr2d.size
data_type=arr2d.dtype

addition=2+arr2d
substraction=arr2d-3
Multiplication=4*arr2d

s=pd.Series([1,2,3,'angel',4,5])

df=pd.DataFrame({'name':['john','alice','Bod'],'age':[30,25,35],'city':['newyork','paris','london']})

df_1=pd.DataFrame({'a':[4,2,6],'b':['m','n','o']})
df_2=pd.DataFrame({'a':[45,8,9],'b':['x','y','z']})

vertical_com=pd.concat([df_1,df_2], axis=0, keys=['df_1','df_2'])

horizontal_com=pd.concat([df_1,df_2], axis=1, keys=['df_1','df_2'])

vertical_com.loc["df_1",0].iloc[0]```


⸻

Screenshots of Results

Include screenshots showing:
	•	NumPy arrays
	•	array operations
	•	Pandas Series output
	•	DataFrame creation
	•	concatenation results

Example:

![NumPy Array Output](screenshots/numpy_array_creation.png)

![Pandas DataFrame](screenshots/pandas_dataframe.png)


⸻

Key Observations

During this lab the following observations were made:
	•	NumPy arrays allow efficient numerical operations.
	•	Vectorized operations make calculations faster than regular Python loops.
	•	Pandas Series provide labeled one-dimensional data structures.
	•	Pandas DataFrames allow structured tabular data manipulation.
	•	Concatenation allows combining multiple DataFrames for analysis.

⸻

Lessons Learned

From this lab, the following skills were gained:
	•	creating NumPy arrays
	•	understanding array properties such as shape and size
	•	performing vectorized numerical operations
	•	creating Pandas Series and DataFrames
	•	accessing data in Pandas objects
	•	combining datasets using concatenation






























