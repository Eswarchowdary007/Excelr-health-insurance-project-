# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:03:26 2019

@author: Gopi
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import Imputer
from scipy.stats import mode
from scipy import stats

insurance=pd.read_csv('file:///C:/Users/Gopi/Desktop/Project-1/Insurance Dataset .csv')

insurance['result'].value_counts() #checking the data is imbalance or not.

insurance.isnull().sum()#checking for the null values


#splitting the categorical data#
insurance['areaservice'].value_counts()
insurance['areaservice']=insurance['areaservice'].fillna(insurance['areaservice'].value_counts().index[0])

insurance['hospitalcounty'].value_counts()
insurance['hospitalcounty']=insurance['hospitalcounty'].fillna(insurance['hospitalcounty'].value_counts().index[0])

insurance['descriptionillness'].value_counts()
insurance['descriptionillness']=insurance['descriptionillness'].fillna(insurance['descriptionillness'].value_counts().index[0])

insurance['Mortalityrisk'].value_counts()
insurance['Mortalityrisk']=insurance['Mortalityrisk'].fillna(insurance['Mortalityrisk'].value_counts().index[0])

insurance['paymenttypology2'].value_counts()
insurance['paymenttypology2']=insurance['paymenttypology2'].fillna(insurance['paymenttypology2'].value_counts().index[0])

insurance['paymenttypology3'].value_counts()
insurance['paymenttypology3']=insurance['paymenttypology3'].fillna(insurance['paymenttypology3'].value_counts().index[0])


insurance.isnull().sum()


mean_value=insurance['certificatenum'].mean()
insurance['certificatenum']=insurance['certificatenum'].fillna(mean_value)
mean_value1=insurance['hospitalid'].mean()
insurance['hospitalid']=insurance['hospitalid'].fillna(mean_value1)

insurance.drop(['zipcode'],inplace=True,axis=1)#have found nan,categorical,numerical values so trying to remove this column
insurance.drop(['weightbaby'],inplace=True,axis=1)#having 0 value
insurance.drop(['yeardischarge'],inplace=True,axis=1)#null values was removed while checking correlation i can see null values 
insurance.drop(['certificatenum'],inplace=True,axis=1)
insurance.drop(['hospitalid'],inplace=True,axis=1)
insurance.drop(['daysspendhospital'],inplace=True,axis=1)

insurance.describe()#checking for the mean median data
insurance.isnull().sum()

ins=insurance.corr()#analyzing correlation values 



from sklearn.preprocessing import LabelEncoder
lb_make=LabelEncoder()
insurance['areaservice']=lb_make.fit_transform(insurance['areaservice'])
insurance['hospitalcounty']=lb_make.fit_transform(insurance['hospitalcounty'])
insurance['hospitalname']=lb_make.fit_transform(insurance['hospitalname'])
insurance['age']=lb_make.fit_transform(insurance['age'])
insurance['gender']=lb_make.fit_transform(insurance['gender'])
insurance['culturalgroup']=lb_make.fit_transform(insurance['culturalgroup'])
insurance['ethnicity']=lb_make.fit_transform(insurance['ethnicity'])
insurance['admissiontype']=lb_make.fit_transform(insurance['admissiontype'])
insurance['homeorselfcare']=lb_make.fit_transform(insurance['homeorselfcare'])
insurance['ccsdiagnosisdescription']=lb_make.fit_transform(insurance['ccsdiagnosisdescription'])
insurance['ccsproceduredescription']=lb_make.fit_transform(insurance['ccsproceduredescription'])
insurance['aprdrgdescription']=lb_make.fit_transform(insurance['aprdrgdescription'])
insurance['aprmdcdescription']=lb_make.fit_transform(insurance['aprmdcdescription'])
insurance['descriptionillness']=lb_make.fit_transform(insurance['descriptionillness'])
insurance['Mortalityrisk']=lb_make.fit_transform(insurance['Mortalityrisk'])
insurance['surgdescription']=lb_make.fit_transform(insurance['surgdescription'])
insurance['paymenttypology1']=lb_make.fit_transform(insurance['paymenttypology1'])
insurance['paymenttypology2']=lb_make.fit_transform(insurance['paymenttypology2'])
insurance['paymenttypology3']=lb_make.fit_transform(insurance['paymenttypology3'])
insurance['abortion']=lb_make.fit_transform(insurance['abortion'])
insurance['emergency_dept']=lb_make.fit_transform(insurance['emergency_dept'])
insurance['result']=lb_make.fit_transform(insurance['result'])


sb.countplot(x='areaservice',data=insurance,palette='hls')
sb.countplot(x='hospitalcounty',data=insurance,palette='hls')
sb.countplot(x='hospitalname',data=insurance,palette='hls')
sb.countplot(x='age',data=insurance,palette='hls')
sb.countplot(x='gender',data=insurance,palette='hls')
sb.countplot(x='culturalgroup',data=insurance,palette='hls')
sb.countplot(x='ethnicity',data=insurance,palette='hls')
sb.countplot(x='admissiontype',data=insurance,palette='hls')
sb.countplot(x='homeorselfcare',data=insurance,palette='hls')
sb.countplot(x='ccsdiagnosisdescription',data=insurance,palette='hls')
sb.countplot(x='ccsproceduredescription',data=insurance,palette='hls')
sb.countplot(x='aprdrgdescription',data=insurance,palette='hls')
sb.countplot(x='aprmdcdescription',data=insurance,palette='hls')
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

plt.hist(insurance.totcharg)

sb.pairplot(insurance)
insurance.describe()#checking for the tendency values 

from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.preprocessing import StandardScaler

std_sca=StandardScaler()#scaling the values 
std_sca.fit(X)

insurance['daysspendhospital']=pd.factorize(insurance.daysspendhospital)[0]
insurance.info()
std_sca=StandardScaler()
X=std_sca.fit_transform(X)


X=insurance.iloc[:,0:27]
Y=insurance.iloc[:,27]

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.20)


classifier=LogisticRegression()
classifier.fit(train_x,train_y)

predict=classifier.predict(test_x)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(test_y,predict))
confusion_matrix(test_y,predict)
insurance.quantile()
