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

NumPy is used for efficient numerical computations and array operations.

## Step 2 – Create a 1D NumPy Array

A one-dimensional array was created from a Python list.
arr1d = np.array([1,2,3,4,5])
arr1d
This produces a simple numeric Step 3 – Create a 2D NumPy Array

## A two-dimensional array was created from a nested list.
arr2d = np.array([[1,2,3],[4,5,6]])
arr2dA 
2D array represents rows and columns similar to a table.

## Step 4 – Create an Array of Zeros
NumPy can generate arrays filled with zeros.
zeros_arr = np.zeros((3,4))
print(zeros_arr)
This creates a 3×4 matrix filled with zeros.

## Step 5 – Check Array Shape

The shape shows the dimensions of the array.
shape = arr2d.shape
shape
Output:(2,3)
Meaning:
	•	2 rows
	•	3 columns
    
## Step 6 – Check Array Size

The size gives the total number of elements in the array.
size = arr2d.size
size
## Step 7 – Check Data Type

NumPy arrays store elements of the same type.
data_type = arr2d.dtype
data_type
## Step 8 – Perform Arithmetic Operations

**Addition**
addition = 2 + arr2d
print(addition)

**Subtraction**
substraction = arr2d - 3
print(substraction)

**Multiplication**
Multiplication = 4 * arr2d
print(Multiplication)














