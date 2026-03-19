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

vertical_com.loc["df_1",0].iloc[0]
```


⸻

Screenshots of Results



<img width="343" height="413" alt="NumPy arrays" src="https://github.com/user-attachments/assets/37b8d15d-6b1e-4a23-81c5-531cb85537cf" />
<img width="839" height="456" alt="Array attributes_1" src="https://github.com/user-attachments/assets/a4a9df7f-cb3f-487d-a324-611a3feaf460" />
<img width="1335" height="297" alt="Array attributes_2" src="https://github.com/user-attachments/assets/4922cdd9-460d-4c29-bde3-85993610c229" />
<img width="587" height="430" alt="image" src="https://github.com/user-attachments/assets/fbe60be2-2aa4-489c-9e7f-256f8835be0d" />
<img width="417" height="359" alt="image" src="https://github.com/user-attachments/assets/45db2537-472b-402c-8e79-424be98e4373" />
<img width="602" height="476" alt="image" src="https://github.com/user-attachments/assets/1186bc31-1586-453f-8a93-dd78ae899c0b" />
<img width="659" height="435" alt="image" src="https://github.com/user-attachments/assets/ee5a04be-13fb-42a7-b130-b554f552fed6" />
<img width="495" height="356" alt="image" src="https://github.com/user-attachments/assets/6a340b9c-7c38-4da5-9d9a-9564957045c1" />
<img width="640" height="450" alt="image" src="https://github.com/user-attachments/assets/80fa7c30-ca56-4334-8f6c-6cbc70539925" />
<img width="406" height="161" alt="image" src="https://github.com/user-attachments/assets/77b94347-f125-4ded-ada7-f1a5925af676" />
<img width="689" height="480" alt="image" src="https://github.com/user-attachments/assets/e57cbd03-1451-4c79-9d0f-943d22cc0f92" />
<img width="431" height="289" alt="image" src="https://github.com/user-attachments/assets/b372e3c9-fdba-4932-9e82-7831979f53a9" />
<img width="443" height="441" alt="image" src="https://github.com/user-attachments/assets/da7405d5-09cc-4727-a4ee-3f444885fffc" />
<img width="612" height="459" alt="image" src="https://github.com/user-attachments/assets/d6b3abe7-4adf-46ec-b31a-268f2f8dbddf" />
<img width="595" height="435" alt="image" src="https://github.com/user-attachments/assets/9a928dd0-b45a-4d58-8341-1a4b2d1f8962" />
<img width="716" height="480" alt="image" src="https://github.com/user-attachments/assets/dd018b2c-4215-4e5d-a9c0-799b65ec8263" />
<img width="595" height="378" alt="image" src="https://github.com/user-attachments/assets/82ea0d07-c122-478c-8388-00509dd9b876" />
<img width="652" height="442" alt="image" src="https://github.com/user-attachments/assets/da697aa9-701b-4040-bc4f-c98a3a5e43c7" />

<img width="573" height="377" alt="image" src="https://github.com/user-attachments/assets/f5c1ceb4-eafc-479e-a30a-8c62d03234d9" />



## Pandas DataFrame SCREENSHOT
<img width="645" height="478" alt="image" src="https://github.com/user-attachments/assets/78db84f4-6dc2-453a-a6d3-e27e9e9a4d91" />

<img width="618" height="432" alt="image" src="https://github.com/user-attachments/assets/3c300acf-b4e1-4291-9fa7-a2634dec757b" />

<img width="613" height="415" alt="image" src="https://github.com/user-attachments/assets/dd907391-ce04-4c23-95b4-8aaa7248d4c1" />

<img width="573" height="429" alt="image" src="https://github.com/user-attachments/assets/33178ffc-9371-439d-b302-7ab1d3a6625a" />

<img width="486" height="302" alt="image" src="https://github.com/user-attachments/assets/8270e150-b387-4389-937f-59a0dcda579b" />

<img width="644" height="363" alt="image" src="https://github.com/user-attachments/assets/2a71a02b-9d36-4639-9124-0cc52451cdc6" />

<img width="602" height="349" alt="image" src="https://github.com/user-attachments/assets/2daafc15-d49d-4e83-b9bf-5c923533ed72" />

<img width="605" height="342" alt="image" src="https://github.com/user-attachments/assets/3e48448f-c7e7-48e5-964b-ecfe4e80b805" />

<img width="617" height="304" alt="image" src="https://github.com/user-attachments/assets/56332999-22ec-4893-bb6f-2ea6cccae993" />

<img width="601" height="325" alt="image" src="https://github.com/user-attachments/assets/9552a748-dd4e-4ae6-9f7d-2050a549af8e" />

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






























