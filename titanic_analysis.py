import pandas as pd


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.dates as matdates
plt.style.use('ggplot')
import math




titanic_df = pd.read_csv('train.csv')
print(titanic_df.head())

print(titanic_df.tail())

print(titanic_df.shape)

print(titanic_df.info())

titanic_df['Age'].hist(bins=10)

def id(passenger):
    age,sex=passenger
    if age<16:
        return 'child'
    else:
        return sex
    
titanic_df['PersonID'] = titanic_df[['Age', 'Sex']].apply(id, axis =1)   
print(titanic_df.head()) 

print(titanic_df['PersonID'].value_counts())

sns.catplot( 'PersonID', data= titanic_df, kind='count')


print(titanic_df['Pclass'].value_counts())
sns.catplot('PersonID', hue = 'Pclass', data = titanic_df, kind = 'count')

print(titanic_df['Embarked'].value_counts())
sns.catplot('PersonID', hue = 'Embarked', data = titanic_df, kind = 'count')


print(titanic_df['SibSp'].value_counts())
sns.catplot('PersonID', hue = 'SibSp', data = titanic_df, kind = 'count')


print(titanic_df['Parch'].value_counts())
sns.catplot('PersonID', hue = 'Parch', data = titanic_df, kind = 'count')


titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'Family'
titanic_df['Alone'].loc[titanic_df['Alone']==0] = 'Alone'

sns.catplot('Alone', data = titanic_df,  kind = 'count')

sns.catplot('Survived', hue = 'Pclass',   data = titanic_df, kind="count")

sns.catplot('Survived', hue = 'PersonID',   data = titanic_df, kind="count")


sns.catplot('Alone',hue='Survived',data=titanic_df,kind="count")


titanic_df.drop("Cabin" , axis = 1 , inplace = True )
print(titanic_df.head())

sex = pd.get_dummies(titanic_df["Sex"] , drop_first = True)
print(sex.head())

embark = pd.get_dummies(titanic_df["Embarked"], drop_first = True)
print(embark.head())


Pcl = pd.get_dummies(titanic_df["Pclass"] , drop_first = True)
print(Pcl.head())

titanic_df = pd.concat([titanic_df , sex , embark , Pcl] , axis = 1)
print(titanic_df.head())

titanic_df.drop(['Sex' , 'Pclass' , 'Embarked' , 'PassengerId' , 'Name' , 'Ticket'] , axis = 1 , inplace = True)
print(titanic_df.head())

titanic_df.drop(['PersonID','Alone'],axis=1,inplace=True)
print(titanic_df.head())

print(titanic_df.isnull().sum())


titanic_df[:] = np.nan_to_num(titanic_df)

X = titanic_df.drop("Survived" , axis =1)
y = titanic_df["Survived"]

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 1)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression() 
logmodel.fit(X_train , y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test , predictions))


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test , predictions))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test , predictions)*100)