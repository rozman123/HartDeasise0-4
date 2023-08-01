import pandas as pd
import torch
from sklearn.model_selection import train_test_split

device=torch.device('cpu')
if torch.cuda.is_available():
    device=torch.device('cuda')



HeartData=pd.read_csv('processed.cleveland.data.csv',sep=',') # we are loading the data
#print(HeartData.head())
X=HeartData.drop(['(num)(prediction)'],axis=1) # we are sellecting the values that will allow us to infer the (num)(prediction)
Y=HeartData['(num)(prediction)'] #what we are predicting

#print(X.head())

Xtrain,Xeval,Ytrain,Yeval=train_test_split(X,Y,train_size=0.25)

Xtrain=torch.tensor(Xtrain.values(),dtype=torch.float32).to(device)
Ytrain=torch.tensor(Ytrain.values(),dtype=torch.float32).to(device)

imput_layer_size=Xtrain.shape[1]
a=70
b=140
model=torch.nn.Sequential(
    torch.nn.Linear(imput_layer_size,a),
    torch.nn.Sigmoid(),
    torch.nn.Linear(a,b),
    torch.nn.Sigmoid(),
    torch.nn.Linear(b,5),
    torch.nn.Sigmoid()
).to(device)

loss=torch.nn.L1Loss()

opt=torch.optim.SGD(Xtrain.parameters(),lr=0.001)

for i in range(0,1001):

    output=model(Xtrain)
    Loss=loss(output,Ytrain.view(-1,1))

    opt.zero_grad()
    Loss.backward()
    opt.step()

    if i%100==0:
        print(i,Loss.item())

model.eval()
with torch.no_grad():
    Xeval=torch.tensor(Xeval.values,dtype=torch.float32).to(device)
    Yeval=torch.tensor(Yeval.values,dtype=torch.float32).to(device)

    model_eval=model(Xeval)
    eval=


