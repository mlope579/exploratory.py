wget https://www.dropbox.com/s/xtms9g31aicstvv/train.csv?dl=1

import pandas as pd

import numpy as np

import matplotlib as plt

%matplotlib inline

plot(arange(5))

df = pd.read_csv("/home/kunal/Downloads/Loan_Prediction/train.csv") #Reading the dataset in a dataframe using Pandas

df.head(10)

df.describe()

df['Property_Area'].value_counts()

df['ApplicantIncome'].hist(bins=50)

df.boxplot(column='ApplicantIncome')

df.boxplot(column='ApplicantIncome', by = 'Education')

df['LoanAmount'].hist(bins=50)

temp1 = df['Credit_History'].value_counts(ascending=True)

temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print ('Frequency Table for Credit History:') 

print (temp1)




print ('\nProbility of getting loan for each Credit History class:')

print (temp2)

ax1 = fig.add_subplot(121)

ax1.set_xlabel('Credit_History')

ax1.set_ylabel('Count of Applicants')

ax1.set_title("Applicants by Credit_History")

temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)

temp2.plot(kind = 'bar')

ax2.set_xlabel('Credit_History')

ax2.set_ylabel('Probability of getting loan')

ax2.set_title("Probability of getting loan by credit history")

 df.apply(lambda x: sum(x.isnull()),axis=0) 
 
 table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
 
 # Define function to return value of this pivot_table
 
 def fage(x): 
 
 return table.loc[x['Self_Employed'],x['Education']]
 
 # Replace missing values
 df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
 
 df['LoanAmount_log'] = np.log(df['LoanAmount'])
 
 df['LoanAmount_log'].hist(bins=20)