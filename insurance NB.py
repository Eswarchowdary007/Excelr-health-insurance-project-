# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:58:01 2019

@author: Gopi
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from scipy.stats import mode
from scipy import stats

insurance=pd.read_csv('file:///C:/Users/Gopi/Desktop/Project-1/Insurance Dataset .csv')


insurance['result'].value_counts() #checking the data is imbalance or not.

insurance.isnull().sum()#checking for the null values

sb.countplot(x='areaservice',data=insurance,palette='hls')
sb.countplot(x='certificatenum',data=insurance,palette='hls')
sb.countplot(x='hospitalcounty',data=insurance,palette='hls')
sb.countplot(x='hospitalid',data=insurance,palette='hls')
sb.countplot(x='hospitalname',data=insurance,palette='hls')
sb.countplot(x='age',data=insurance,palette='hls')
sb.countplot(x='zipcode',data=insurance,palette='hls')
sb.countplot(x='gender',data=insurance,palette='hls')
sb.countplot(x='culturalgroup',data=insurance,palette='hls')
sb.countplot(x='ethnicity',data=insurance,palette='hls')
sb.countplot(x='daysspendhospital',data=insurance,palette='hls')
sb.countplot(x='admissiontype',data=insurance,palette='hls')
sb.countplot(x='homeorselfcare',data=insurance,palette='hls')
sb.countplot(x='yeardischarge',data=insurance,palette='hls')
sb.countplot(x='ccsdiagnosiscode',data=insurance,palette='hls')
sb.countplot(x='ccsdiagnosisdescription',data=insurance,palette='hls')
sb.countplot(x='ccsprocedurecode',data=insurance,palette='hls')
sb.countplot(x='ccsproceduredescription',data=insurance,palette='hls')
sb.countplot(x='aprdrgdescription',data=insurance,palette='hls')
sb.countplot(x='aprmdcdescription',data=insurance,palette='hls')
sb.countplot(x='Codeillness',data=insurance,palette='hls')
sb.countplot(x='descriptionillness',data=insurance,palette='hls')
sb.countplot(x='Mortalityrisk',data=insurance,palette='hls')
sb.countplot(x='surgdescription',data=insurance,palette='hls')
sb.countplot(x='paymenttypology1',data=insurance,palette='hls')
sb.countplot(x='paymenttypology2',data=insurance,palette='hls')
sb.countplot(x='paymenttypology3',data=insurance,palette='hls')
sb.countplot(x='abortion',data=insurance,palette='hls')
sb.countplot(x='emergency_dept',data=insurance,palette='hls')


pd.crosstab(insurance.areaservice,insurance.result).plot(kind='bar')
pd.crosstab(insurance.hospitalcounty,insurance.result).plot(kind='bar')
pd.crosstab(insurance.hospitalname,insurance.result).plot(kind='bar')
pd.crosstab(insurance.age,insurance.result).plot(kind='bar')
pd.crosstab(insurance.gender,insurance.result).plot(kind='bar')
pd.crosstab(insurance.culturalgroup,insurance.result).plot(kind='bar')
pd.crosstab(insurance.ethnicity,insurance.result).plot(kind='bar')
pd.crosstab(insurance.admissiontype,insurance.result).plot(kind='bar')
pd.crosstab(insurance.homeorselfcare,insurance.result).plot(kind='bar')
pd.crosstab(insurance.ccsdiagnosisdescription,insurance.result).plot(kind='bar')
pd.crosstab(insurance.ccsproceduredescription,insurance.result).plot(kind='bar')
pd.crosstab(insurance.aprdrgdescription,insurance.result).plot(kind='bar')
pd.crosstab(insurance.aprmdcdescription,insurance.result).plot(kind='bar')
pd.crosstab(insurance.descriptionillness,insurance.result).plot(kind='bar')
pd.crosstab(insurance.Mortalityrisk,insurance.result).plot(kind='bar')
pd.crosstab(insurance.surgdescription,insurance.result).plot(kind='bar')
pd.crosstab(insurance.paymenttypology1,insurance.result).plot(kind='bar')
pd.crosstab(insurance.paymenttypology2,insurance.result).plot(kind='bar')
pd.crosstab(insurance.paymenttypology3,insurance.result).plot(kind='bar')
pd.crosstab(insurance.abortion,insurance.result).plot(kind='bar')
pd.crosstab(insurance.emergency_dept,insurance.result).plot(kind='bar')

insurance['Mortalityrisk'].value_counts()
insurance['Mortalityrisk']=insurance['Mortalityrisk'].fillna(insurance['Mortalityrisk'].value_counts().index[0])


ins_columns=['age','hospitalname','gender','culturalgroup','admissiontype','homeorselfcare','ccsproceduredescription',
                'aprdrgdescription','aprmdcdescription','Mortalityrisk','surgdescription','paymenttypology1',
                'emergency_dept']

insurance.dtypes
insurance.daysspendhospital.unique()
insurance.daysspendhospital.replace("120 +",120,inplace=True)

for i in ins_columns:
    number = preprocessing.LabelEncoder()
    insurance[i] = number.fit_transform(insurance[i])

sb.boxplot(x='areaservice',y='result',data=insurance,palette='hls')
sb.boxplot(x='hospitalcounty',y='result',data=insurance,palette='hls')
sb.boxplot(x='hospitalname',y='result',data=insurance,palette='hls')
sb.boxplot(x='age',y='result',data=insurance,palette='hls')
sb.boxplot(x='gender',y='result',data=insurance,palette='hls')
sb.boxplot(x='culturalgroup',y='result',data=insurance,palette='hls')
sb.boxplot(x='ethnicity',y='result',data=insurance,palette='hls')
sb.boxplot(x='admissiontype',y='result',data=insurance,palette='hls')
sb.boxplot(x='homeorselfcare',y='result',data=insurance,palette='hls')
sb.boxplot(x='ccsdiagnosisdescription',y='result',data=insurance,palette='hls')
sb.boxplot(x='ccsproceduredescription',y='result',data=insurance,palette='hls')
sb.boxplot(x='aprdrgdescription',y='result',data=insurance,palette='hls')
sb.boxplot(x='aprmdcdescription',y='result',data=insurance,palette='hls')
sb.boxplot(x='descriptionillness',y='result',data=insurance,palette='hls')
sb.boxplot(x='Mortalityrisk',y='result',data=insurance,palette='hls')
sb.boxplot(x='surgdescription',y='result',data=insurance,palette='hls')
sb.boxplot(x='paymenttypology1',y='result',data=insurance,palette='hls')
sb.boxplot(x='paymenttypology2',y='result',data=insurance,palette='hls')
sb.boxplot(x='paymenttypology3',y='result',data=insurance,palette='hls')
sb.boxplot(x='abortion',y='result',data=insurance,palette='hls')
sb.boxplot(x='emergency_dept',y='result',data=insurance,palette='hls')
#splitting the categorical data#


insurance.isnull().sum()
insurance.skew()
insurance.kurt()

insurance.result.value_counts()

insurance.drop(['zipcode'],inplace=True,axis=1)#have found nan,categorical,numerical values so trying to remove this column
insurance.drop(['weightbaby'],inplace=True,axis=1)#having 0 value
insurance.drop(['yeardischarge'],inplace=True,axis=1)#null values was removed while checking correlation i can see null values 
insurance.drop(['certificatenum'],inplace=True,axis=1)
insurance.drop(['hospitalid'],inplace=True,axis=1)
insurance.drop(['areaservice'],inplace=True,axis=1)
insurance.drop(['hospitalcounty'],inplace=True,axis=1)
insurance.drop(['descriptionillness'],inplace=True,axis=1)
insurance.drop(['paymenttypology2'],inplace=True,axis=1)
insurance.drop(['paymenttypology3'],inplace=True,axis=1)
insurance.drop(['abortion'],inplace=True,axis=1)
insurance.drop(['ethnicity'],inplace=True,axis=1)
insurance.drop(['ccsprocedurecode'],inplace=True,axis=1)
insurance.drop(['ccsdiagnosiscode'],inplace=True,axis=1)
insurance.drop(['ccsdiagnosisdescription'],inplace=True,axis=1)
insurance.drop(['totcharg'],inplace=True,axis=1)


insurance.describe()#checking for the mean median data
insurance.isnull().sum()

ins=insurance.corr()#analyzing correlation values 


from sklearn.preprocessing import LabelEncoder
lb_make=LabelEncoder()
insurance['result']=lb_make.fit_transform(insurance['result'])

plt.hist(insurance.totcharg)

insurance.describe()#checking for the tendency values 

X=insurance.iloc[:,0:17]
Y=insurance.iloc[:,17]

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.30)
gnb=GaussianNB()
gnb.fit(train_x,train_y)

y_pred=gnb.predict(test_x)

from sklearn.metrics import accuracy_score
accuracy_score(test_y,y_pred) #74.62 accuracy

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(test_y,y_pred))

confusion_matrix(test_y,y_pred)

#ADA BOOST CLASSIFIER
ada=AdaBoostClassifier()
ada.fit(train_x,train_y)

ada_pred=ada.predict(test_x)
accuracy_score(test_y,ada_pred) #74.95 accuracy

print(classification_report(test_y,ada_pred))
confusion_matrix(test_y,ada_pred)

#GRADIENT BOOST CLASSIFIER
gd=GradientBoostingClassifier()
gd.fit(train_x,train_y)

gd_pred=gd.predict(test_x)
accuracy_score(test_y,gd_pred) #75.10 accuracy

print(confusion_matrix(test_y,gd_pred))

#GRADIENT BOOST PARAMETER TUNING
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3)
ins=GradientBoostingClassifier(learning_rate=0.01,n_estimators=100,max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10)
gbs=GradientBoostingClassifier()
parameters={'n_estimators':[50,100,500,800,1000],'max_depth':[1,3,5,7,10],'learning_rate':[0.01,0.1,1,10,100]}

from sklearn.model_selection import GridSearchCV
cv=GridSearchCV(gbs,parameters,cv=5)
cv.fit(X,Y.values.ravel())

