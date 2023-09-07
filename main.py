import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
# About the data: https://archive.ics.uci.edu/dataset/45/heart+disease
# Additional Information:
# This database contains 76 attributes, but all published experiments refer
# to using a subset of 14 of them.  In particular, the Cleveland database is
# the only one that has been used by ML researchers to date.
# The "goal" field refers to the presence of heart disease in the patient.
# It is integer valued from 0 (no presence) to 4. Experiments
# with the Cleveland database have concentrated on simply attempting
# to distinguish presence (values 1,2,3,4) from absence (value 0).
'''

device=torch.device('cpu')
if torch.cuda.is_available():
    device=torch.device('cuda')


data=pd.read_csv('processed.cleveland.data.csv',sep=',',na_values=['?']) # we are loading the data

HeartData=data.dropna()

X=HeartData.drop(['(num)(prediction)'],axis=1) # we are sellecting the values that will allow us to infer the (num)(prediction)
Y=HeartData['(num)(prediction)'] #what we are predicting

Ylisttrain=[]
for i in Y:
    if i == 0:
        Ylisttrain.append(0)
    else:
        Ylisttrain.append(1)


Y=torch.tensor(Ylisttrain,dtype=torch.float32).to(device)

Xtrain_,Xeval_,Ytrain_,Yeval_=train_test_split(X,Y,train_size=0.80)

scalar = StandardScaler()   # standarisation
Xtrain=scalar.fit_transform(Xtrain_)
Xeval=scalar.transform(Xeval_)

Xtrain=torch.tensor(Xtrain,dtype=torch.float32).to(device)
Ytrain=Ytrain_.clone().detach().to(device)



imput_layer_size=Xtrain.shape[1]
a=100
b=200
model=torch.nn.Sequential(
    torch.nn.Linear(imput_layer_size,a),
    torch.nn.Sigmoid(),
    torch.nn.Linear(a,b),
    torch.nn.Sigmoid(),
    torch.nn.Linear(b,1),
    torch.nn.Sigmoid()
).to(device)


loss=torch.nn.BCELoss()

opt=torch.optim.Adam(model.parameters(),lr=0.001)


for epoch in range(0,1501):

    output=model(Xtrain)
    # output=torch.sigmoid(output)
    Loss=loss(output, Ytrain.view(-1,1))

    opt.zero_grad()
    Loss.backward()
    opt.step()

    if epoch%100==0:
        print(epoch,f'Error: {Loss.item():.4f}')
# print(Ytrain.view(-1, 5))
model.eval()

with torch.no_grad():

    Xeval=torch.tensor(Xeval,dtype=torch.float32).to(device)

    model_eval=model(Xeval)

    #predictions=torch.argmax(model_eval,dim=1)
    predictions=[]
    for i in model_eval:
        if i >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    predictions=torch.tensor(predictions,dtype=torch.float32).to(device)
    #predictions=torch.round(model_eval)
    print(predictions)
    efficiency=(Yeval_.to(device)==predictions).float().mean()

    print(f'Skuteczność {efficiency.item():.4f}')

