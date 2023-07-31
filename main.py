import pandas as pd
import torch
from sklearn.model_selection import train_test_split

urzadzenie=torch.device('cpu')
if torch.cuda.is_available():
    urzadzenie=torch.device('cuda')



HeartData=pd.read_csv('processed.cleveland.data.csv',sep=',') # we are loading the data
#print(HeartData.head())
X=HeartData.drop(['(num)(prediction)'],axis=1) # we are sellecting the values that will allow us to infer the (num)(prediction)
Y=HeartData['(num)(prediction)'] #what we are predicting

#print(X.head())

Xtrain,Xeval,Y_train,Yeval=train_test_split(X,Y,train_size=0.25)



