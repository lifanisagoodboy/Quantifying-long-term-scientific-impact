from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

#dta=[5,36,49,27,42,47,37,32,38,25,32,27,21,18,19,24,13,19,19,19,22,25,16,21]
#dta=[27,53,54,103,102,121,133,139,173,130,130,138,88,106,86,96,69,80,78]

#dta=[162,206,153,274,257,184,120,133,343,411,633,752,829,1139,1073,1959,2734,
#3552,3932,4053,3113,4118,5188,4864,6159,3655,6523,6071,4479,1942,1085,1552,
#1027,3946,8796,10943,14348,18542,21446,17709,19520,27296,24005,21410,24637,
#24484,18942,21760,25920,28006,29331,37634,39567,38011,40663,40521,45327]

dta=[2,13,15,16,20,6,15,9,8,6,15,9,4,8,3,2,4,8,7,10,16,12,12,15,15,
     8,10,13,9,14,13,24,10,12,14,16,20,15]

'''
dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422, 
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355, 
10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767, 
12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232, 
13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248, 
9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722, 
11999,9390,13481,14795,15845,15271,14686,11054,10395]
'''

dta=np.array(dta,dtype=np.float)

dta=pd.Series(dta)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1978','2015'))
dta.plot(figsize=(12,8),color='black',marker = "*",linewidth=3)
plt.savefig(u'数据原始时序图.png')
#plt.show()

fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)
diff1 = dta.diff(1)
print (u"一阶差分后的数据值:")
print (diff1)
diff1.plot(ax=ax1,color = 'black',marker = "o",linewidth=3)
plt.savefig(u'数据一阶差分后时序图.png')
#plt.show()

#fig, ax = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True)
#fig, ax = plt.subplots(2)


fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta,lags=35,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta,lags=35,ax=ax2)
plt.savefig(u'平稳时间序列的自相关图和偏自相关图.png')
#plt.show()

print (u"信息量输出")

arma_mod70 = sm.tsa.ARMA(dta,(1,0)).fit()
print(arma_mod70.aic,arma_mod70.bic,arma_mod70.hqic)
arma_mod30 = sm.tsa.ARMA(dta,(0,1)).fit()
print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
arma_mod71 = sm.tsa.ARMA(dta,(1,1)).fit()
print(arma_mod71.aic,arma_mod71.bic,arma_mod71.hqic)
arma_mod80 = sm.tsa.ARMA(dta,(2,0)).fit()
print(arma_mod80.aic,arma_mod80.bic,arma_mod80.hqic)

print (u"输出残差")
resid = arma_mod70.resid
#print (type(resid))
print (resid)


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=35, ax=ax2)
plt.savefig(u'残差的ACF图和PACF图.png')
#plt.show()

print (u"进一步进行D-W检验")
print (sm.stats.durbin_watson(arma_mod70.resid.values))

print(stats.normaltest(resid))
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.savefig(u'QQ图.png')
#plt.show()

print (u"残差序列Ljung-Box检验,Q检验")
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,38), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

print (u"模型预测")
predict_dta = arma_mod70.predict('2015', '2020', dynamic=True)
print(predict_dta)

fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.ix['2000':].plot(ax=ax)
fig = arma_mod70.plot_predict('2015', '2020', dynamic=True, ax=ax, plot_insample=False)

plt.savefig('test.png')
#plt.show()
