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

Ylisttrain=[]
for i in Y:
    if i == 1:
        Ylisttrain.append([0, 1, 0, 0, 0])
    elif i == 2:
        Ylisttrain.append([0, 0, 1, 0, 0])
    elif i == 3:
        Ylisttrain.append([0, 0, 0, 1, 0])
    elif i == 4:
        Ylisttrain.append([0, 0, 0, 0, 1])
    else:
        Ylisttrain.append([1, 0, 0, 0, 0])

#Ylisttrain=[torch.tensor(i,dtype=torch.float32).to(device) for i in Ylisttrain]
#Y=torch.stack(Ylisttrain)
#Ylisttrain=[i.clone().detach().requires_grad_(True).to(device) for i in Ylisttrain]
Y=torch.tensor(Ylisttrain).to(device)
#Y=Ylisttrain.clone().detach().requires_grad_(True).to(device)
#print(X.head())

Xtrain_,Xeval_,Ytrain_,Yeval_=train_test_split(X,Y,train_size=0.80)
#print(Xtrain_)
#print(Ytrain_)
#
scalar = StandardScaler()   # standarisation
Xtrain=scalar.fit_transform(Xtrain_)
Xeval=scalar.transform(Xeval_)

Xtrain=torch.tensor(Xtrain,dtype=torch.float32).to(device)
Ytrain=Ytrain_.clone().detach().to(device)
#Ytrain=torch.tensor(Ytrain_,dtype=torch.float32).to(device)
#print(Ytrain)



imput_layer_size=Xtrain.shape[1]
a=100
b=200
model=torch.nn.Sequential(
    torch.nn.Linear(imput_layer_size,a),
    torch.nn.ReLU(),
    torch.nn.Linear(a,b),
    torch.nn.ReLU(),
    torch.nn.Linear(b,5)
).to(device)

loss=torch.nn.CrossEntropyLoss()

opt=torch.optim.SGD(model.parameters(),lr=0.001)


for i in range(0,5001):

    output=model(Xtrain)
    #if i ==2500:
         #print(output)
    Loss=loss(output, Ytrain.argmax(dim=1))

    opt.zero_grad()
    Loss.backward()
    opt.step()

    if i%100==0:
        print(i,f'Error: {Loss.item():.4f}')
# print(Ytrain.view(-1, 5))
model.eval()

with torch.no_grad():

    Xeval=torch.tensor(Xeval,dtype=torch.float32).to(device)
    #Yeval=torch.tensor(Yeval_,dtype=torch.float32).to(device)

    model_eval=model(Xeval)
    #print(model_eval)
    # model_list=[]
    # for i in model_eval:
    #     if i[1] >= 0.5:
    #         model_list.append([0,1,0,0,0])
    #     elif i[2] >= 0.5:
    #         model_list.append([0,0,1,0,0])
    #     elif i[3] >= 0.5:
    #         model_list.append([0,0,0,1,0])
    #     elif i[4] >= 0.5:
    #         model_list.append([0,0,0,0,1])
    #     else:
    #         model_list.append([1,0,0,0,0])

    #model_list=[torch.tensor(i,dtype=torch.float32).to(device) for i in model_list]
    #model_list=torch.stack(model_list)
    #model_list=[i.clone().detach().requires_grad_(True).to(device) for i in model_list ]
    #model_eval=torch.tensor(model_list,dtype=torch.float32).to(device)
    #model_eval=model_eval.clone().requires_grad_(True).to(device)
    predictions=torch.argmax(model_eval,dim=1)
    print(predictions)
    efficiency=(Yeval_.to(device).argmax(dim=1)==predictions).float().mean()

    print(f'Skuteczność {efficiency.item():.4f}')


