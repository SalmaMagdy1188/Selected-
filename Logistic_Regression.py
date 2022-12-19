#!/usr/bin/env python
# coding: utf-8

# In[72]:


# loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder


# In[73]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
print (bc.feature_names)
print (bc.target_names)


# In[74]:


bc = load_breast_cancer()

bc_df = pd.DataFrame(bc.data, columns=bc.feature_names)
target_df = pd.DataFrame(bc.target).rename(columns={0:'Diagnosis'})

dataset_df = pd.concat([bc_df, target_df], axis=1)

dataset_df.head()


# In[75]:


def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred

nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(nums, sigmoid(nums), 'r')


# In[76]:


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression(lr=0.01)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print('Accuracy: ',acc)


# In[77]:


## plot a ROC Curve
def Snippet_140_Ex_2():
    
    from sklearn.linear_model import LogisticRegression
    import warnings 
    get_ipython().run_line_magic('matplotlib', 'inline')
    warnings.filterwarnings("ignore")

    # Creating feature matrix and target vector
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target


    # Spliting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Creating classifier
    clf_tree = DecisionTreeClassifier(); clf_reg = LogisticRegression();

    # Training model
    clf_tree.fit(X_train, y_train); clf_reg.fit(X_train, y_train);

    # Getting predicted probabilities
    y_score1 = clf_tree.predict_proba(X_test)[:,1]
    y_score2 = clf_reg.predict_proba(X_test)[:,1]

    # Ploting Receiving Operating Characteristic Curve
    # Creating true and false positive rates
    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)
    false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_score2)
    print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_score2))


    plt.subplots(1, figsize=(5,5))
    plt.title('Receiver Operating Characteristic - Logistic regression')
    plt.plot(false_positive_rate2, true_positive_rate2)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

Snippet_140_Ex_2()


# In[78]:


confusion_martrix=confusion_matrix(y_test, y_pred)
print (confusion_martrix)


# In[88]:


def deconfusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    
    cm = {
        'Predicted (Positive)': [tp, fp],
        'Predicted (Negative)': [fn, tn],
    }

    df = pd.DataFrame(cm, columns = ['Predicted (Positive)', 'Predicted (Negative)'], 
                      index=['Actual (Positive)', 'Actual (Negative)'])
    
    return df
    
dcm = deconfusion_matrix(y_test, y_pred)
dcm


# In[89]:


tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()


# In[90]:


from sklearn.metrics import precision_score, recall_score, f1_score 

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)


# In[79]:


import seaborn as sns
plt.figure(figsize=(5,5))
sns.heatmap(confusion_martrix, annot=True, fmt='d', cmap='YlGnBu')
plt.ylabel("Actual Digits")
plt.xlabel("Recgonize Digits")


# In[80]:


p=np.arange(0,1,0.01)


# In[81]:


plt.plot(p,-1 *np.log(p))
plt.show()


# In[82]:


plt.plot(p,-1 * (np.log(1-p)))
plt.show()


# In[83]:


plt.plot(p,-1 * np.log(p))
plt.plot(p,-1 * (np.log(1-p)), 'r')
plt.show()

