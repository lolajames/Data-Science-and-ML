
# Lab 2: Python Functions

## Objective
The objective of this lab is to understand how functions work in Python. Functions help organize code into reusable blocks that perform specific tasks.

## Tools Used
- Python
- Google Colab
- GitHub

## Topics Covered
- Python Functions
- Function Definition
- Function Parameters
- Return Statement
- Lambda Functions
- map() Function
- filter() Function

## Step-by-Step Process

1. Define functions using the `def` keyword.
2. Pass parameters to functions.
3. Use the `return` statement to return values.
4. Create anonymous functions using `lambda`.
5. Apply functions using `map()` and `filter()`.

## Commands Executed
### Function Definition
#### funtion to greet the user
def greet(name):
  """Greets the user"""
  return f'Hello,{name}!'
### calling the function
message=greet('Alice')
print(message)
output: Hello,Alice!

### Return Vaule
def add(a,b):
    """This code adds two numbers."""
    c= a+b
    return c
add(8,7)
output: 15

def subt(a,b):
  d=b-a
  return d
  
subt(3,9)
outtput : 6

### Function to return the sum and product of two numbers
def sum_and_product(a,b):
    return a+b,a*b
result=sum_and_product(3,4)
print('sum:',result[0])
print('product:',result[1])
sum: 7
product: 12
### Lambda Functions:
#### adding the numbers
add=lambda x,y:x+y
print(add(9,5))
output : 14

students= [{'name':'ololade','score':87},
           {'name':'moses','score':78}]
sorted_students=sorted(students,key=lambda x: x['name'],
                        reverse=True)
print(sorted_students)
Output : [{'name': 'ololade', 'score': 87}, {'name': 'moses', 'score': 78}]

### map(function, iterable):

#### finding the square of the numbers in the list
numbers=[1,2,3,4,5]
squared=map(lambda x: x**2,numbers)
print(list(squared))
output : [1, 4, 9, 16, 25]

#### converting temperature from celsius to fhrenheit using map:
temps_celsius =[0,10,20,30,40]
temps_fhrenheit=map(lambda c:(c*9/5)+32,temps_celsius)
print(list(temps_fhrenheit))
output: [32.0, 50.0, 68.0, 86.0, 104.0]

#### Capitalizing the first letter of each word in a list of strings using map:
words=['hello','world','python']
capitalized_words=map(lambda w:w.capitalize(),words)
print(list(capitalized_words))
Output : ['Hello', 'World', 'Python']
### filter(function, iterable):
#### finding the even numbers in the list
numbers=[1,2,3,4,5]
evens=filter(lambda x:x%2==0,numbers)
print(list(evens))
output: [2, 4]

#### Filtering out negative numbers from a list:
numbers=[-2,5,-7,3,-1,8]
positive_numbers=filter(lambda x: x>=0, numbers)
print(list(positive_numbers))
output: [5, 3, 8]

#### Filtering out even numbers from a list:
numbers=[1,2,3,4,5,6,7,8,9,10]
even_numbers=filter(lambda x: x%2==0,numbers)
print(list(even_numbers))
output: [2, 4, 6, 8, 10]
### reduce(function, iterable[, initializer])
#### finding the product of the elements in the list
from functools import reduce
numbers=[1,2,3,4,5]
product =reduce(lambda x,y:x*y,numbers)
print(product)
output: 120
#### Finding the maximum element in a list using reduce:
numbers=[3,8,1,6,5]
max_num=reduce(lambda x,y:x if x>y else y,numbers)
print(max_num)
output: 8
### sum(iterable[, start]):
#### Returns the sum of all elements in the iterable.
#find the sum of the numbers in the list
numbers=[1,2,3,4,5]
total=sum(numbers)
print(total)
output : 15
#### max(iterable[, key, default]):
Returns the maximum element from the iterable.
numbers=[1,2,3,4,5]
max_num= max(numbers)
print(max_num)
output: 5
#### Finding the longest string in a list:
strings=['apple','banana','orange','kiwi']
longest_string= max(strings, key=len)
print(longest_string)
output : banana

#### min(iterable[, key, default]):
Returns the minimum element from the iterable.
#### Finding the minimum value in a list
numbers=[1,2,3,4,5]
min_num=min(numbers)
print(min_num)
output: 1

#### Finding the minimum value in a dictionary:
scores = {"Alice": 85, "Bob": 90, "Charlie": 75}
min_score=min(scores.values())
print(min_score)
output: 75

## Observations

Functions in Python help organize code into smaller, reusable blocks that perform specific tasks.

Using the def keyword allows programmers to define custom functions that can be called multiple times within a program.

Functions with parameters make the code more flexible because different values can be passed when the function is called.


The return statement allows a function to send results back to the part of the program that called it.

Lambda functions provide a concise way to write simple functions without formally defining them using def.

The map() function is useful for applying a function to every element in a list, making data transformation easier.

The filter() function helps select elements from a list that meet specific conditions.

These concepts demonstrate how Python supports efficient and structured programming, which is important when working with data processing tasks.

## Lessons Learned

I learned how to define and use functions in Python using the def keyword.

I practiced passing parameters to functions and using the return statement to return values from a function.

I gained a better understanding of how functions help make code more organized, reusable, and easier to maintain.

I explored lambda functions and learned how they can be used to write short, anonymous functions in a more concise way.

I practiced using the map() function to apply a function to each element in a list.

I learned how the filter() function can be used to select elements from a list based on a condition.

This lab helped strengthen my understanding of Python programming concepts that are important for data manipulation and analysis in data science.

