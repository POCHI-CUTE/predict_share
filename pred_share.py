# coding: utf-8

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

mydf1 = pd.read_csv("NIKKEI20152020.csv", index_col = 0, parse_dates=[0])
mydf2 = pd.read_csv("NY.csv", index_col = 0, parse_dates=[0])
mydf3 = pd.read_csv("USDJPY.csv", index_col = 0, parse_dates=[0])
mydf = pd.merge(mydf1, mydf2, on = "日付")
mydf = pd.merge(mydf, mydf3, on = "日付")
mydf = mydf.sort_index()
N = len(mydf)
nikkei = mydf["Nikkei終値"]
nikkei = nikkei.apply(lambda x: x.replace(",", "")).astype(np.float)
NY = mydf["NY終値"]
NY = NY.apply(lambda x: x.replace(",", "")).astype(np.float)
usdjpy = mydf["USDJPY終値"]

L = 365
W = 4
newdf = pd.DataFrame(index=mydf.index, columns=[])
Y = nikkei[W - 1:N]
for i in range(1, W):
    newdf["x" + str(i)] = nikkei.shift(i).T
newdf["NY"] = NY.shift(1).T
newdf["USDJPY"] = usdjpy.shift(1).T
X = newdf[W - 1:N]

model = linear_model.LinearRegression()

M = 570
Ytest = np.empty(M)
Ypred = np.empty(M)
Ydate = []
for m in range(M):
    xtrain = X[m:m + L]
    ytrain = Y[m:m + L]
    model.fit(xtrain, ytrain)

    Ytest[m] = Y[m + L:m + L + 1]
    ydate = str(Y.index[m + L]).split()[0]
    Ydate.append(ydate)
    xtest = X[m + L:m + L + 1]
    ypred = model.predict(xtest)
    Ypred[m] = ypred[0]
    print("ydate=", Ydate, "ytest=", Ytest[m], "ypred=", Ypred[m])

mse = metrics.mean_squared_error(Ytest, Ypred)
print("MSE=", mse)
mae = np.sum(np.abs(Ypred - Ytest)) / M
print("MAE=", mae)
msre = np.dot((Ypred - Ytest) / Ytest, (Ypred - Ytest) / Ytest) / M
print("MSRE=", msre)
mare = np.sum(np.abs(Ypred - Ytest) / Ytest) / M
print("MARE=", mare)

plt.title("Prediction of Nikkei (Price)")
plt.xlabel("date")
plt.ylabel("price")
plt.grid()
plt.plot(Ytest, color="blue")
plt.plot(Ypred, color="red")
plt.show()