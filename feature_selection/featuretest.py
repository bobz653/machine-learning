#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
example learning 
reference :
http://scikit-learn.org/stable/modules/feature_selection.html
http://www.tuicool.com/articles/ieUvaq
'''
import numpy as np
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

#classification
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#分类
iris = load_iris()
X, y = iris.data, iris.target
print X.shape
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print X_new.shape


#Load boston housing dataset as an example
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

#random forest feature selection 也存在关联特征打分不稳定
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     #count every colum
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), names[i]))
print "RF model:",sorted(scores, reverse=True)

rf.fit(X, Y)
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True)

#LR feature selection  当特征有关联的时候，不稳定 比如x1=x2 =>y =x1+x2 ,y=-x1+3x2
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)
#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
  if names == None:
    names = ["X%s" % x for x in range(len(coefs))]
  lst = zip(coefs, names)
  if sort:
    lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
  return " + ".join("%s * %s" % (round(coef, 3), name)
                   for coef, name in lst)
print "Linear model:", pretty_print_linear(lr.coef_,names,True)


#LR with L1    当特征有关联的时候，不稳定 比如x1=x2 =>y =x1+x2 ,y=-x1+3x2 L1 惩罚都为2 
#LR with L2    这里0系数就少了，对关联特征惩罚不同 上面 一个是2 一个就是10了，对L2来各个系数相等惩罚最小，所以稳定
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

scaler = StandardScaler()
X = scaler.fit_transform(boston["data"])
Y = boston["target"]
names = boston["feature_names"]
lasso = Lasso(alpha=.3)
lasso.fit(X, Y)
print "Lasso model: ", pretty_print_linear(lasso.coef_, names, sort = True)

ridge = Ridge(alpha=10)
ridge.fit(X,Y)
print "Ridge model: ", pretty_print_linear(ridge.coef_, names, sort = True)

#random lasso 克服纯lasso和随机森林不稳定方法
from sklearn.linear_model import RandomizedLasso
#using the Boston housing data. 
#Data gets scaled automatically by sklearn's implementation
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

rlasso = RandomizedLasso(alpha=0.025)
rlasso.fit(X, Y)

print "Features sorted by RandomizedLasso:"
print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), 
                 names), reverse=True)
