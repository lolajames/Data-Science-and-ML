
# Lab 4: Python Strings and Control Flow

## Objective
The objective of this lab is to understand how Python handles string operations and control flow. This lab demonstrates how to manipulate text data using Python built-in string methods.

## Tools Used
- Python
- Google Colab
- GitHub

## Step-by-Step Process

1. Open Google Colab notebook
2. Define a string variable
3. Apply various Python string methods
4. Execute commands and observe outputs

## Commands Executed

### Convert string to uppercase
python
name = 'ParoCyber LLC'
name.upper()
### Convert string to uppercase
python
name.lower()
### Check if string contains only alphabets
name.isalpha()
### Split sentence into words
sentence = "Parocyber is the best"
sentence.split(" ")
### Join words with a separator
words = sentence.split(" ")
"-".join(words)
### Replace word 
name = "ParoCyberLLC"
name_ = name.replace('LLC','Institute')
print(name_)
### Startswith
name = "ParoCyberLLC"
print(name.startswith("P"))
### endswith()
name = "ParoCyberLLC"
print(name.endswith('C'))

## Observations
Python provides several built-in functions for manipulating text.

String methods like upper() and lower() help standardize text.

Functions like split() and join() are useful for processing textual data.
## Lessons Learned

String manipulation is an important step in data preprocessing.

Practicing Python syntax helps avoid common errors.

Understanding built-in functions improves coding efficiency.
