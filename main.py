import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device=torch.device('cpu')
if torch.cuda.is_available():
    device=torch.device('cuda')



data=pd.read_csv('processed.cleveland.data.csv',sep=',',na_values=['?']) # we are loading the data
#print(HeartData.head())
HeartData=data.dropna()
#print(HeartData)
X=HeartData.drop(['(num)(prediction)'],axis=1) # we are sellecting the values that will allow us to infer the (num)(prediction)
Y=HeartData['(num)(prediction)'] #what we are predicting

#print(X.head())

Xtrain_,Xeval_,Ytrain_,Yeval_=train_test_split(X,Y,train_size=0.25)
#print(Xtrain_)
#print(Ytrain_)
#
scalar = StandardScaler()   # standarisation
Xtrain=scalar.fit_transform(Xtrain_)
Xeval=scalar.transform(Xeval_)

Xtrain=torch.tensor(Xtrain,dtype=torch.float32).to(device)
Ytrain=torch.tensor(Ytrain_.values,dtype=torch.float32).to(device)
#print(Ytrain)



imput_layer_size=Xtrain.shape[1]
a=100
b=240
model=torch.nn.Sequential(
    torch.nn.Linear(imput_layer_size,a),
    torch.nn.Sigmoid(),
    torch.nn.Linear(a,b),
    torch.nn.ReLU(),
    torch.nn.Linear(b,1),
    torch.nn.Sigmoid()
).to(device)

loss=torch.nn.BCELoss()# CrossEntropyLoss better for classification problems

opt=torch.optim.SGD(model.parameters(),lr=0.01)

for i in range(0,2501):

    output=model(Xtrain)
    #if i ==2500:
         #print(output)
    Loss=loss(output,Ytrain.view(-1,1))

    opt.zero_grad()
    Loss.backward()
    opt.step()

    if i%100==0:
        print(i,f'Error: {Loss.item():.4f}')

model.eval()
with torch.no_grad():

    Xeval=torch.tensor(Xeval,dtype=torch.float32).to(device)
    Yeval=torch.tensor(Yeval_.values,dtype=torch.long).to(device)


    model_eval=model(Xeval)
    #print(model_eval)


    #print(model_eval)
    #print(Yeval)
    efficiency=(Yeval==model_eval).float().mean()

    print(f'Skuteczność {efficiency.item():.4f}')


