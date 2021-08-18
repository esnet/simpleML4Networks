import pandas as pd
import json
import urllib2
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np

#r = urllib2.urlopen('http://fnal-pt1.es.net:8085/esmond/perfsonar/archive/0019bdd444af4340b505d1d4cfb3d79e/histogram-owdelay/statistics/3600')
#r = urllib2.urlopen('http://fnal-pt1.es.net:8085/esmond/perfsonar/archive/0019bdd444af4340b505d1d4cfb3d79e/packet-loss-rate/aggregations/3600')
#####
r = urllib2.urlopen('http://graphite.es.net/snmp/west/fnal-mr2/interface/xe-8_1_0.5/in?begin=1466935200&end=1472331600')
#r = urllib2.urlopen('http://graphite.es.net/snmp/west/fnal-mr2/interface/xe-8_1_0.5/in?begin=1466935200')

j = json.load(r)

l = {}
n = {'timestamp':[], 'values':[]}

##### Replace old dict with l and delete the rest of the stuff to get back to where we originally were 
old = {}
old = {key:value for key, value in dict(j['data']).iteritems()} 
#baseCount = 30
baseCount = 600
for k,v in sorted(old.iteritems()):
    baseCount -= 1
    if baseCount == 0:
        l[k] = v
        baseCount = 30
        

#for d in j:
    #l[d['ts']] = d['val']['mean']
    #l[d['ts']] = d['val']

for i in sorted(l.iteritems()):
    n['timestamp'].append(i[0])
    n['values'].append(i[1])

data = pd.DataFrame(n)

data['timestamp'] = pd.to_datetime(data['timestamp'], unit = 's')
data.set_index('timestamp', inplace = True)

data1 = data.iloc[:1430, :]
data2 = data.iloc[1430:1630,:]

#data1 = data.iloc[:3888, :]
#data2 = data.iloc[3888:,:]

print data.size
print data1.size
print data2.size

ts1 = data1['values']
ts2 = data2['values']
print ts1
print ts2

def test_stationarity(timeseries):
    rolmean = pd.rolling_mean(timeseries, window = 12)
    rolstd = pd.rolling_std(timeseries, window = 12)
    print "results of fickey fuller test"
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags used', 'Number of observations used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)'%key] = value
    print dfoutput


#test_stationarity(data['values'])


#Differencing - Might use it
ts_log = np.log(ts1)
#ts_log_diff = ts_log - ts_log.shift()
#plt.plot(ts_log_diff)
#plt.show()
#ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)
ts_log.dropna(inplace=True)
test_stationarity(ts_log)

'''
#Decomposing - Just for fun. Won't be using it.
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
ts_decompose = residual
ts_decompose.dropna(inplace=True)
test_stationarity(ts_decompose)
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
'''

lag_acf = acf(ts_log, nlags = 20)
lag_pacf = pacf(ts_log, nlags = 20, method = 'ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

plt.show()

#from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
#model = ARIMA(ts_log, order = (2,0,6))
#model = SARIMAX(ts_log, order = (2,0,4), seasonal_order = (2,0,4,24))
model = SARIMAX(ts_log, order = (4,0,6), seasonal_order = (4,0,6,24))
#model = SARIMAX(ts_log, order = (1,0,1), seasonal_order = (1,0,1,24))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log)
plt.plot(results_ARIMA.fittedvalues, color='r')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log)**2))
plt.show()

#results_ARIMA.save('model2.pkl')


#predictions_ARIMA_log = pd.Series(results_ARIMA.fittedvalues, copy=True)
#print predictions_ARIMA_log

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)


#Can be removed if no differencing has been performed
#predictions_ARIMA_diff_cumsum = predictions_ARIMA.cumsum()
#predictions_ARIMA_log = pd.Series(ts_log.ix[0], index = ts_log.index)
#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts1, color = 'b',label = 'ts1 (not in log)')
plt.plot(predictions_ARIMA, color = 'r', label ='predictions of known values (not in log)' )
plt.legend()
plt.show()

print "Time for predictions"

future = pd.Series(results_ARIMA.predict(start = '2017-01-21 09:33:30', end = '2017-01-23 11:33:30'), copy= True)
#future = results_ARIMA.forecast(325)
#print future[:][0]
#print "Next"
#print future[:][1]
#future = pd.Series(results_ARIMA.forecast()[0],copy=True)
print future
plt.plot(future, color = 'r', label = 'future predictions in log')
plt.plot(np.log(ts2), color = 'b', label = 'actual future values in log')
plt.legend()
plt.show()

plt.plot(np.exp(future), color = 'r', label = 'future predictions (not in log)')
plt.plot(ts2, color = 'b', label = 'actual future values (not in log)')
plt.legend()
plt.show()
