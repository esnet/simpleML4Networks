import pandas as pd
import json
#import urllib2
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA

r = pd.read_csv('datasets/2019snmp_concat_data_morefeatures.csv')
#r = urllib2.urlopen('http://graphite.es.net/snmp/west/fnal-mr2/interface/xe-8_1_0.5/in?begin=1466935200')




data = pd.DataFrame(r)
data= data['SACR_SUNN_in']
data.plot()
plt.title('SACR_SUNN')
plt.show()

data1 = data.iloc[:8000]
data2 = data.iloc[8000:8738]


print(data.size)
print(data1.size)
print(data2.size)

ts1 = data1
ts2 = data2
print(ts1)
print(ts2)


def test_stationarity(timeseries):
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    print("results of fickey fuller test")
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags used', 'Number of observations used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)'%key] = value
    print(dfoutput)


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



#
model = ARIMA(ts_log, order = (5,1,0))
results_ARIMA = model.fit()
print(results_ARIMA.summary())
plt.plot(ts_log)
plt.plot(results_ARIMA.fittedvalues, color='r')
plt.title('Learning dominant trends- SUNN-SACR')
#plt.show()

#results_ARIMA.save('model2.pkl')



predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)


#Can be removed if no differencing has been performed
#predictions_ARIMA_diff_cumsum = predictions_ARIMA.cumsum()
#predictions_ARIMA_log = pd.Series(ts_log.ix[0], index = ts_log.index)
#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts1, color = 'b',label = 'Real Data')
plt.plot(predictions_ARIMA, color = 'r', label ='Predicted Data' )
plt.legend()
plt.show()
