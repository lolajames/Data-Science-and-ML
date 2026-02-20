## Lab Title
# Data Structures — List Operations in Python
# Objective

The objective of this lab is to understand Python data structures, focusing on:

1. Creating lists

2. Modifying lists using built-in methods

3. Adding elements to a list

4. Removing elements

5. Finding element positions using indexing

   This lab strengthens understanding of data manipulation
   # Tools Used
   Google Colab
   ## Step-by-Step Process
   # Creating a List
   We created a list containing letters:
   Letters = ["a","b","c","d","e"]
Letters
Output:
['a', 'b', 'c', 'd', 'e']
# Using append()
The append() method adds a single item to the end of the list.

Letters.append("f")
print(Letters)

Output:

['a','b','c','d','e','f']
# Using extend()
The extend() method adds multiple elements to the list.

Letters.extend(["g","h"])
print(Letters)

Output:

['a','b','c','d','e','f','g','h']
# Using insert()
insert() adds an element at a specific index.

Letters.insert(5,"X")
print(Letters)

Output:

['a','b','c','d','e','X','f','g','h']
# Using remove()
remove() deletes a specific element.

Letters.remove("X")
print(Letters)

Output:

['a','b','c','d','e','f','g','h']
# Finding Index Position
We used index() to find the position of an element.

Index = Letters.index("e")
print(Index)

Output:

4
# Commands Executed
   append()

extend()

insert()

remove()

index()
# Key Observations

Lists are dynamic and easy to modify.

append() adds one item; extend() adds multiple items.

insert() allows precise positioning.

remove() deletes by value.

index() helps locate elements quickly.
