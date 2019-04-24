# lplplpk.github.io
my blog
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


#留出法,分层抽样
       
def hold_out(dataname) :   
    train_data = pd.read_csv(dataname)
    # 打印信息
    #sns.distplot(train_data.Age,fit=stats.gamma,kde=False)
    #sns.plt.show()
    #train_data.info()
    #train_data
    
    #print('*'*50)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Survived']
    features1=['Pclass','Age','SibSp','Parch','Fare']
    x_train = train_data[features]
    x_train.info()
    #print ('*'*50)
    x_train['Age'].fillna(x_train['Age'].mean(),inplace=True)
    #print (x_train['Embarked'].value_counts())
    x_train['Embarked'].fillna('S',inplace=True)
    #rint(train_data['Survived'].value_counts())
    #用热编码解决
    X_train=x_train.sort_values(by='Survived')
    dummies1=pd.get_dummies(X_train['Sex'],prefix='Sex')
    dummies2=pd.get_dummies(X_train['Embarked'],prefix='Embarked')
    dummies3=pd.get_dummies(X_train['Survived'],prefix='Survived')
    dummies=((X_train[features1].join(dummies1)).join(dummies2))
 
    m1=np.array(train_data['Survived'].value_counts())[0]
    m2=np.array(train_data['Survived'].value_counts())[1]           
    k=int(m1*0.7)
    x_train1=dummies.iloc[:k,:]
    x_train2=dummies.iloc[m1+int(m2*0.7):(m1+m2),:]
    x_train=np.array(pd.concat([x_train1,x_train2],ignore_index=True))
    x_test=np.array(dummies.iloc[k:m1+int(m2*0.7),:])
    y_trainlabel1=dummies3.iloc[:k,:]
    y_trainlabel2=dummies3.iloc[m1+int(m2*0.7):(m1+m2),:]
    y_trainlabel=np.array(pd.concat([y_trainlabel1,y_trainlabel2],ignore_index=True))
    y_testlabel=np.array(dummies3.iloc[k:m1+int(m2*0.7 ),:])
    return x_train,y_trainlabel,x_test,y_testlabel
    
