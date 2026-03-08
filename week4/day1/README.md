
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
## String Formatting
name = "John"
age = 25
print("My name is {} and I am {} years old".format(name, age))
Output: My name is John and I am 25 years old
### Using f-strings (Formatted String Literals):
name = "John"
age = 25

print(f"My name is {name} and I am {age} years old")
### Using the .format() method
name = "Clifford"
age = 45
print("Hello, my name is {} and I am {} years old.".format(name,age))

name = "Euodia"
age = 25
print("Hello, my name is {1} and I am {0} years old.".format(name,age))

## Control Flow: If Statements, Loops, and Conditional Statements
### If Statements
#if statement
x = 10
if x > 0 and x % 2 == 0:
    print("x is an even number")
    output: x is an even number
y = 15
if y < 0 and y % 2 == 1:
  print("y is an odd number")
  output: y is an odd number
### If-Else Statements
x = int(input("Enter a number: "))
if x > 0 and x % 2 == 0:
    print("x is an even number")
else:
    print("x is an odd number")

  x = 15
if x > 0 and x % 2 == 0:
    print("x is an even number")
else:
    print("x is an odd number")
output: x is an odd number

x = int(input("Enter a number: "))
if x > 0 and x% 2 == 0:
    print("x is an even number")
else:
    print("x is an odd number")

### Nested if

    if age >= 18:
    if age < 21:
        print('You are an adult, but not yet allowed to drink')
    else:
        print("You are an adult and allowed to drink")
else:
    print("You are underage")
output: You are underage

### If-Elif-Else Statements

age = 30
if age >= 18 and age < 21:
    print('You are an adult but not allowed to drink yet')
elif age >= 18 and age >=21:
    print("You are an adult and allowed to drink")
elif age >= 18 or age < 21:
    print("I am adult")
else:
    print("you are underage")
output: You are an adult and allowed to drink

score = int(input('Enter a score: '))
if score >= 90:
    print('A')
elif score >= 80:
    print('B')
elif score >= 70:
    print('C')
else:
    print('Fail')
## Loops
### iterating over a list
list_ = ['apple','banana','orange','mango']
for item in list_:
    print(item)
### iterating over a list to print odd numbers
list_ = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for i in list_:
    if i % 2 == 1:
        print(i)
### iterating over a dictionary
person = {'name': 'Clifford','age': 28,'city':'Saskatchewan'}
for key, value in person.items():
    print(f"{key}:{value}")
### iterating to get sum
list_ = [20,30,40,50,60,70,80]
total = 0
for item in list_:
    total += item
print(total)
## While Loops
###  Looping and printing a value
i = 1
while i <= 10:
    print(i)
    i += 1
### looping over a list and printing the items in the list
  emma = ['Apple','Banana', 'Cashew']
i = 0
while i <= 2:
    print(emma[i])
    i +=1
### Looping over a dictionary and printing each key-value pair
my_dict = {'a': 1, 'b': 2, 'c': 3}
keys = list(my_dict.keys())  # Convert dictionary keys to list for indexing
index = 0
while index < len(keys):
    key = keys[index]
    value = my_dict[key]
    print(key, value)
    index += 1
### Calculating the sum of numbers in a list using a while loop

numbers = [1, 2, 3, 4, 5]
total = 0
index = 0
while index < len(numbers):
    total += numbers[index]
    index += 1
print("Sum of numbers:", total)


## Observations
Python provides many built-in string methods that make text manipulation simple and efficient.

Methods such as upper() and lower() are useful for standardizing text data.

Functions like split() and join() help convert between strings and lists, which is helpful when processing textual data.

String formatting using .format() and f-strings allows dynamic insertion of variables into strings, making output more readable and structured.

Membership operators like in help check whether a substring exists within a string.

Control flow statements such as if, elif, and else allow programs to make decisions based on conditions.

Loops like for and while enable repeated execution of code, which is useful for processing multiple items or running iterative tasks.

Understanding these basic Python concepts is important because they form the foundation for data manipulation and preprocessing in data science.
## Lessons Learned

I learned how to use Python string methods such as upper(), lower(), split(), and join() to manipulate and process text data.

I gained a better understanding of how string formatting works using .format() and f-strings, which helps create dynamic and readable output.

I practiced using conditional statements (if, elif, else) to control how a program behaves based on certain conditions.

I learned how loops (for and while) can be used to repeat actions and process data efficiently.

I improved my understanding of basic Python syntax and debugging errors, which is important when writing and running code.


