# --------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Code starts here
df=pd.read_csv(filepath_or_buffer=path, compression='zip', low_memory = False)
print(df.columns)
X=df.iloc[:,:]
y=df['loan_status']
X.drop(columns=['loan_status'],axis=1,inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=4)

# Code ends here


# --------------
# Code starts  here
col=df.isnull().sum() * 100 / len(df)
col_drop=[]
for i in col.index:
    if col[i]>25:
        col_drop.append(i)
# print(col_drop)
for j in col.index:
    if X[j].nunique()==1:
        col_drop.append(j)
# print(col_drop)
X_train.drop(columns=col_drop,axis=1,inplace=True)
X_test.drop(columns=col_drop,axis=1,inplace=True)
# Code ends here


# --------------
import numpy as np


# Code starts here
y_train=np.where((y_train == 'Fully Paid') |(y_train == 'Current'), 0, 1)
y_test=np.where((y_test == 'Fully Paid') |(y_test == 'Current'), 0, 1)
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder


# categorical and numerical variables
cat = X_train.select_dtypes(include = 'O').columns.tolist()
num = X_train.select_dtypes(exclude = 'O').columns.tolist()

# Filling missing values

# Train Data

for x in cat:
    mode = X_train[x].mode()[0]
    X_train[x].fillna(mode, inplace = True)

for x in num:
    mean = X_train[x].mean()
    X_train[x].fillna(mean,inplace = True)

# Test Data
for x in cat:
    mode = X_train[x].mode()[0]
    X_test[x].fillna(mode,inplace = True)

for x in num:
    mean = X_train[x].mean()
    X_test[x].fillna(mean,inplace = True)


# Label encoding

le = LabelEncoder()
for x in cat:
    
    X_train[x] = le.fit_transform(X_train[x])
    X_test[x] = le.fit_transform(X_test[x])


# --------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix,classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

# Code starts here
rf=RandomForestClassifier(random_state =42,max_depth=2, min_samples_leaf=5000)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
f1=f1_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
roc_auc=roc_auc_score(y_test,y_pred)
print('Confusion matrix:',confusion_matrix(y_test,y_pred))
print('classification report:',classification_report(y_test,y_pred))
y_pred_proba=rf.predict_proba(X_test)[:,1]
fpr,tpr,_=metrics.roc_curve(y_pred,y_test)
auc=metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Random Forest model, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
# Code ends here


# --------------
from xgboost import XGBClassifier

# Code starts here
xgb=XGBClassifier(learning_rate=0.0001)
xgb.fit(X_train,y_train)
y_pred=xgb.predict(X_test)
f1=f1_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
roc_auc=roc_auc_score(y_test,y_pred)
print('Confusion matrix:',confusion_matrix(y_test,y_pred))
print('classification report:',classification_report(y_test,y_pred))
y_pred_proba=rf.predict_proba(X_test)[:,1]
fpr,tpr,_=metrics.roc_curve(y_pred,y_test)
auc=metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="XGBoost model, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
# Code ends here


