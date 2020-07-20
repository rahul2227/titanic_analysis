# Titanic_Analysis
Titanic_analysis

# Project Aimed:
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this Model, I have tried to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

# Pre-Requisites:
Python3 
Some of the Python Libraries
1.  numpy           "https://numpy.org/"
2.  pandas          "https://pypi.org/project/pandas/"
3.  seaborn         "https://pypi.org/project/seaborn/"
4.  sklearn         "https://pypi.org/project/sklearn/"
5.  matplotlib      "https://pypi.org/project/matplotlib/"

# Installation:
There are some steps for installation
1.  Install Python3.                        "https://www.python.org/downloads/"
2.  Download and Install Anaconda Toolkit   "https://www.anaconda.com/products/individual"  
3.  Install Spyder.                         
4.  Install all the libraries above mentioned by using "pip install library_name".
5.  Download the project. Run it in your system.

# Dataset:
In the given dataset there are following columns such as gender, age, name, etc. In the dataset, those who survived are mentioned with a label "!" and those who are not with"0".

Note: The dataset is downloaded from Kaggle: "https://www.kaggle.com/c/titanic".

I used the two datasets train and test for training and testing my model respectively.

# Algorithm:
As the data is structured and after analyzing the dataset I came to know that we can use classifier in this project. So the classifier used in this is Logistic Regression. In this seaborn library that uses matplotlib is used to plot graphs and pandas is used for reading the dataset. 
I am getting 80% accuracy in this project through Logistic Regression.

