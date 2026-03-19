
# Lab Title
## Introduction to Data Visualization Using Matplotlib

---

# Objective

The objective of this lab is to reproduce classroom exercises demonstrating the fundamentals of **data visualization using Matplotlib**.

The lab focuses on:

- Understanding the purpose of data visualization
- Importing and using the Matplotlib library
- Creating basic plots using `matplotlib.pyplot`
- Visualizing datasets using charts
- Understanding how visualizations help interpret data
- Displaying graphical representations of numerical data

This lab helps build foundational skills required for **data visualization and exploratory data analysis in Data Science and Machine Learning**.

---

# Tools Used

The following tools and technologies were used during this lab:

- **Python** – programming language used for the exercises  
- **Matplotlib** – library used for creating data visualizations  
- **NumPy** – library used for numerical computations  
- **Pandas** – library used for data manipulation and analysis  
- **Jupyter Notebook / Google Colab** – environment used to write and run the code  
- **GitHub** – used for documentation and version control  

---

# Step-by-Step Process

---

# Step 1 – Import Required Libraries

The first step is to import the required Python libraries.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```python

Explanation:

* `matplotlib.pyplot` is used for creating plots and charts.
* `numpy` is used for numerical computations.
* `pandas` is used for handling structured datasets.

---

# Step 2 – Understanding Matplotlib

Matplotlib is a Python library used for **data visualization**. It allows us to create graphical representations such as:

* line charts
* bar charts
* scatter plots
* histograms
* pie charts

Visualization helps make data easier to understand and analyze.

---

# Step 3 – Creating Data for Visualization

Before creating a plot, we need some data to visualize.

Example data:

```python
x = [1,2,3,4,5]
y = [10,20,25,30,40]
```

Here:

* `x` represents the horizontal axis values
* `y` represents the vertical axis values

---

# Step 4 – Create a Line Plot

A line chart can be created using the `plot()` function.

```python
plt.plot(x, y)
plt.show()
```

This generates a **simple line graph** that shows the relationship between the x and y values.

---

# Step 5 – Add Chart Labels

Labels help make the chart easier to understand.

```python
plt.plot(x, y)

plt.title("Simple Line Chart")
plt.xlabel("X Values")
plt.ylabel("Y Values")

plt.show()
```

Explanation:

* `title()` adds a title to the graph
* `xlabel()` labels the horizontal axis
* `ylabel()` labels the vertical axis

---

# Step 6 – Create a Bar Chart

Bar charts are useful for comparing categories.

```python
categories = ["A","B","C","D"]
values = [5,7,3,8]

plt.bar(categories, values)
plt.show()
```

This creates a **bar chart representing categorical data**.

---

# Step 7 – Create a Scatter Plot

Scatter plots show relationships between two numerical variables.

```python
plt.scatter(x, y)
plt.show()
```

Scatter plots are commonly used in **data exploration and correlation analysis**.

---
# Commands Executed


Below are the commands executed during the Matplotlib lab.

```python
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the Breast Cancer dataset
bc = load_breast_cancer()
bc

# Extract features and labels
x = bc.data
y = bc.target
feature_names = bc.feature_names
target_names = bc.target_names

# Create a DataFrame for easier exploration
df = pd.DataFrame(x, columns=feature_names)
df['target'] = pd.Categorical.from_codes(y, target_names)

# Display first 5 rows
print(df.head())
````

---

```python
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
iris = load_iris()
iris
```

---

```python
# Extract features and labels
x = iris.data
y = iris.target
target_names = iris.target_names

# Bar Plot
plt.figure(figsize=(8,6))
unique, counts = np.unique(y, return_counts=True)

plt.bar(
    target_names,
    counts,
    color=plt.cm.viridis(np.linspace(0,1,len(target_names)))
)

plt.title('Bar Plot of Iris Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()
```

---

```python
# Check dataset shape
x.shape
```

---

```python
# Pie Plot
plt.figure(figsize=(8,6))
plt.pie(
    np.bincount(y),
    labels=target_names,
    autopct='%1.1f%%',
    startangle=120
)

plt.title('Pie Plot of Iris Species')
plt.show()
```

---

```python
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
iris = load_iris()
x = iris.data
y = iris.target
target_names = iris.target_names

# Heatmap (Correlation Matrix)
plt.figure(figsize=(8,6))

corr_matrix = np.corrcoef(x.T)

plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')

# Add annotations
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        plt.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center')

# Add colorbar
plt.colorbar(label='Correlation Coefficient')

# Labels
plt.xticks(ticks=np.arange(len(iris.feature_names)), labels=iris.feature_names)
plt.yticks(ticks=np.arange(len(iris.feature_names)), labels=iris.feature_names)

plt.title('Heatmap of Correlation Matrix (Iris Dataset)')
plt.show()
```

---

```python
# Line Plot
plt.figure(figsize=(8,6))

plt.plot(x[:,0], label=iris.feature_names[0])

plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()

plt.show()
```

---

```python
# Scatter Plot
plt.figure(figsize=(8,6))

for i in range(len(target_names)):
    plt.scatter(x[y==i,0], x[y==i,1], label=target_names[i])

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.legend()

plt.show()
```

---

```python
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Box Plot
plt.figure(figsize=(8,6))

colors = plt.cm.viridis(np.linspace(0,1,len(target_names)))

for i, species in enumerate(target_names):
    plt.boxplot(
        X[y == i, 2],
        positions=[i],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor=colors[i])
    )

plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.title('Box Plot of Petal Length by Species')
plt.xticks(ticks=np.arange(len(target_names)), labels=target_names)

plt.show()
```

---

```python
# Violin Plot
plt.figure(figsize=(8,6))

for i, species in enumerate(target_names):
    plt.violinplot(
        X[y == i, 3],
        positions=[i],
        widths=0.6,
        showmeans=False,
        showmedians=True
    )

plt.xlabel('Species')
plt.ylabel('Petal Width (cm)')
plt.title('Violin Plot of Petal Width by Species')
plt.xticks(ticks=np.arange(len(target_names)), labels=target_names)

plt.show()
```

---

```python
# Histogram
plt.figure(figsize=(8,6))

plt.hist(
    X[:,0],
    bins=20,
    alpha=0.6,
    color='cyan',
    label='Sepal Length'
)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Sepal Length')

plt.legend()

plt.savefig('histogram.png')

plt.show()
```

```





# Screenshots of Results


* Bar chart output
* Scatter plot output
*  Line chart output

```
![Line Chart](screenshots/line_chart.png)

![Bar Chart](screenshots/bar_chart.png)

![Scatter Plot](screenshots/scatter_plot.png)
```

---

# Key Observations

During this lab the following observations were made:

* Matplotlib allows easy creation of different types of charts.
* Visualization helps make data easier to interpret.
* Graphical representations can reveal patterns and relationships in data.
* Proper labels and titles improve the readability of charts.

---

# Lessons Learned

From this lab, the following skills were gained:

* importing and using the Matplotlib library
* creating line charts, bar charts, and scatter plots
* labeling charts for better readability
* visualizing data to support analysis
* understanding the role of visualization in Data Science workflows

These skills are essential for **exploratory data analysis and presenting insights from data**.

```

---

### After adding this, your **Week 5 folder should now look like this**

```

week5
├── README.md
│
├── day1-numpy-pandas
│   ├── numpy_lab.ipynb
│   ├── pandas_lab.ipynb
│   └── README.md
│
└── day2-matplotlib
├── matplotlib_lab.ipynb
└── README.md

```

---

✅ This now **matches the same documentation process** used in your NumPy/Pandas lab:

- Lab Title  
- Objective  
- Tools Used  
- Step-by-Step Process  
- Commands Executed  
- Screenshots  
- Observations  
- Lessons Learned  

Which means it satisfies the **assignment rubric**.

---

If you want, I can also show you **one small trick that will make your entire GitHub repository look like a professional Data Scientist portfolio instead of just a class assignment.**
```

