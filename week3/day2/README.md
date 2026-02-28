## Lab Title: Python Strings and String Methods
# Objective
The objective of this lab is to understand string data types in Python and practice commonly used string methods for modifying, analyzing, and manipulating text data.
# Tools Used
Jupyter Notebook (Google Colab)
GitHub for documentation
## Step-by-Step Process
# Step 1: Creating a String
Name_ = "ParoCyber LLC"
Name_
ParoCyber LLC
# Step 2: Converting to Lowercase
Name_ = Name_.lower()
Name_
parocyber llc
# Step 3: Converting to Uppercase
Name_=Name_.upper()
Name_
PAROCYBER LLC
# Step 4: Checking String Content
Name_.isalpha()   # Checks if all characters are alphabets
Name_.isdigit()   # Checks if all characters are digits
Name_.islower()   # Checks if all characters are lowercase
Name_.isupper()   # Checks if all characters are uppercase
# Step 5: Splitting a String
Flower = "Last week the singles didn't get flowers"
Flower.split(" ")
# Step 6: Joining Strings
word = Flower.split(" ")
"_".join(word)
# Step 7: Replacing Words
Flower.replace("singles", "men")
# Step 8: Checking Starting Characters
Flower = "roses"
Flower.startswith("r")
# Step 9: Checking Ending Characters
Flower2 = "sunflower"
Flower2.endswith("r")
# Commands Executed
All string methods were executed in Python using built-in string functions such as:

upper()

lower()

isalpha()

isdigit()

split()

join()

replace()

startswith()

endswith()
# Key Observations / Lessons Learned

Strings are immutable in Python.

String methods return new values rather than modifying the original string.

Methods like split() and join() are useful for text processing.

replace() is useful for modifying specific parts of text.

startswith() and endswith() help in pattern checking.

String manipulation is fundamental for data cleaning and NLP tasks in data science.
