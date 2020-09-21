import matplotlib.pyplot as plt  
import pandas as pd  
from fbprophet import Prophet  
  
# Read the dataset from CSVs  
df_AirHoliday = pd.read_csv('Air_Holiday_forProphet.csv')  
print(df_AirHoliday.head())  
train_df_AirHoliday = df_AirHoliday[df_AirHoliday['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_AirHoliday = df_AirHoliday[df_AirHoliday['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_AirBusiness = pd.read_csv('Air_Business_forProphet.csv')  
print(df_AirBusiness.head())  
train_df_AirBusiness = df_AirBusiness[df_AirBusiness['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_AirBusiness = df_AirBusiness[df_AirBusiness['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_AirVFR = pd.read_csv('Air_VFR_forProphet.csv')  
print(df_AirVFR.head())  
train_df_AirVFR = df_AirVFR[df_AirVFR['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_AirVFR = df_AirVFR[df_AirVFR['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_AirStudy = pd.read_csv('Air_Study_forProphet.csv')  
print(df_AirStudy.head())  
train_df_AirStudy = df_AirStudy[df_AirStudy['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_AirStudy = df_AirStudy[df_AirStudy['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_AirMiscellaneous = pd.read_csv('Air_Miscellaneous_forProphet.csv')  
print(df_AirMiscellaneous.head())  
train_df_AirMiscellaneous = df_AirMiscellaneous[df_AirMiscellaneous['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_AirMiscellaneous = df_AirMiscellaneous[df_AirMiscellaneous['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
  
df_SeaHoliday = pd.read_csv('Sea_Holiday_forProphet.csv')  
print(df_SeaHoliday.head())  
train_df_SeaHoliday = df_SeaHoliday[df_SeaHoliday['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_SeaHoliday = df_SeaHoliday[df_SeaHoliday['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_SeaBusiness = pd.read_csv('Sea_Business_forProphet.csv')  
print(df_SeaBusiness.head())  
train_df_SeaBusiness = df_SeaBusiness[df_SeaBusiness['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_SeaBusiness = df_SeaBusiness[df_SeaBusiness['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_SeaVFR = pd.read_csv('Sea_VFR_forProphet.csv')  
print(df_SeaVFR.head())  
train_df_SeaVFR = df_SeaVFR[df_SeaVFR['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_SeaVFR = df_SeaVFR[df_SeaVFR['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_SeaStudy = pd.read_csv('Sea_Study_forProphet.csv')  
print(df_SeaStudy.head())  
train_df_SeaStudy = df_SeaStudy[df_SeaStudy['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_SeaStudy = df_SeaStudy[df_SeaStudy['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_SeaMiscellaneous = pd.read_csv('Sea_Miscellaneous_forProphet.csv')  
print(df_SeaMiscellaneous.head())  
train_df_SeaMiscellaneous = df_SeaMiscellaneous[df_SeaMiscellaneous['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_SeaMiscellaneous = df_SeaMiscellaneous[df_SeaMiscellaneous['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
  
df_TunnelHoliday = pd.read_csv('Tunnel_Holiday_forProphet.csv')  
print(df_TunnelHoliday.head())  
train_df_TunnelHoliday = df_TunnelHoliday[df_TunnelHoliday['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_TunnelHoliday = df_TunnelHoliday[df_TunnelHoliday['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_TunnelBusiness = pd.read_csv('Tunnel_Business_forProphet.csv')  
print(df_TunnelBusiness.head())  
train_df_TunnelBusiness = df_TunnelBusiness[df_TunnelBusiness['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_TunnelBusiness = df_TunnelBusiness[df_TunnelBusiness['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_TunnelVFR = pd.read_csv('Tunnel_VFR_forProphet.csv')  
print(df_TunnelVFR.head())  
train_df_TunnelVFR = df_TunnelVFR[df_TunnelVFR['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_TunnelVFR = df_TunnelVFR[df_TunnelVFR['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_TunnelStudy = pd.read_csv('Tunnel_Study_forProphet.csv')  
print(df_TunnelStudy.head())  
train_df_TunnelStudy = df_TunnelStudy[df_TunnelStudy['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_TunnelStudy = df_TunnelStudy[df_TunnelStudy['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
df_TunnelMiscellaneous = pd.read_csv('Tunnel_Miscellaneous_forProphet.csv')  
print(df_TunnelMiscellaneous.head())  
train_df_TunnelMiscellaneous = df_TunnelMiscellaneous[df_TunnelMiscellaneous['ds'] < '2019-01-01']  # Use data before 2019 to train the model  
test_df_TunnelMiscellaneous = df_TunnelMiscellaneous[df_TunnelMiscellaneous['ds'] >= '2018-12-31']  # Use 2019 data to test and predict 2020 data  
  
# Instantiating Prophet object  
m_AH = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_AH.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_AH.fit(train_df_AirHoliday)  
m_AB = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_AB.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_AB.fit(train_df_AirBusiness)  
m_AV = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_AV.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_AV.fit(train_df_AirVFR)  
m_AS = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_AS.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_AS.fit(train_df_AirStudy)  
m_AM = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_AM.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_AM.fit(train_df_AirMiscellaneous)  
  
m_SH = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_SH.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_SH.fit(train_df_SeaHoliday)  
m_SB = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_SB.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_SB.fit(train_df_SeaBusiness)  
m_SV = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_SV.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_SV.fit(train_df_SeaVFR)  
m_SS = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_SS.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_SS.fit(train_df_SeaStudy)  
m_SM = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_SM.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_SM.fit(train_df_SeaMiscellaneous)  
  
m_TH = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_TH.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_TH.fit(train_df_TunnelHoliday)  
m_TB = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_TB.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_TB.fit(train_df_TunnelBusiness)  
m_TV = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_TV.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_TV.fit(train_df_TunnelVFR)  
m_TS = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_TS.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_TS.fit(train_df_TunnelStudy)  
m_TM = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=2)  
m_TM.add_seasonality(name='quarterly', period=91.5, fourier_order=7, prior_scale=0.02)  
m_TM.fit(train_df_TunnelMiscellaneous)  
  
# Predict a future dataframe by quarter  
future_AirHoliday = m_AH.make_future_dataframe(periods=len(test_df_AirHoliday), freq='Q')  
print(future_AirHoliday.tail())  
future_AirBusiness = m_AB.make_future_dataframe(periods=len(test_df_AirBusiness), freq='Q')  
print(future_AirBusiness.tail())  
future_AirVFR = m_AV.make_future_dataframe(periods=len(test_df_AirVFR), freq='Q')  
print(future_AirVFR.tail())  
future_AirStudy = m_AS.make_future_dataframe(periods=len(test_df_AirStudy), freq='Q')  
print(future_AirStudy.tail())  
future_AirMiscellaneous = m_AM.make_future_dataframe(periods=len(test_df_AirMiscellaneous), freq='Q')  
print(future_AirMiscellaneous.tail())  
  
future_SeaHoliday = m_SH.make_future_dataframe(periods=len(test_df_SeaHoliday), freq='Q')  
print(future_SeaHoliday.tail())  
future_SeaBusiness = m_SB.make_future_dataframe(periods=len(test_df_SeaBusiness), freq='Q')  
print(future_SeaBusiness.tail())  
future_SeaVFR = m_SV.make_future_dataframe(periods=len(test_df_SeaVFR), freq='Q')  
print(future_SeaVFR.tail())  
future_SeaStudy = m_SS.make_future_dataframe(periods=len(test_df_SeaStudy), freq='Q')  
print(future_SeaStudy.tail())  
future_SeaMiscellaneous = m_SM.make_future_dataframe(periods=len(test_df_SeaMiscellaneous), freq='Q')  
print(future_SeaMiscellaneous.tail())  
  
future_TunnelHoliday = m_TH.make_future_dataframe(periods=len(test_df_TunnelHoliday), freq='Q')  
print(future_TunnelHoliday.tail())  
future_TunnelBusiness = m_TB.make_future_dataframe(periods=len(test_df_TunnelBusiness), freq='Q')  
print(future_TunnelBusiness.tail())  
future_TunnelVFR = m_TV.make_future_dataframe(periods=len(test_df_TunnelVFR), freq='Q')  
print(future_TunnelVFR.tail())  
future_TunnelStudy = m_TS.make_future_dataframe(periods=len(test_df_TunnelStudy), freq='Q')  
print(future_TunnelStudy.tail())  
future_TunnelMiscellaneous = m_TM.make_future_dataframe(periods=len(test_df_TunnelMiscellaneous), freq='Q')  
print(future_TunnelMiscellaneous.tail())  
  
# Create a new forecast dataframe and display table  
forecast_AirHoliday = m_AH.predict(future_AirHoliday)  
print(forecast_AirHoliday[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_AirHoliday.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_AirHoliday.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_AirHoliday.yhat_upper.sum()))  
sum_y_AH = 0  
sum_yhat_AH = 0  
for x in range(0, 71):  
     sum_y_AH = sum_y_AH + df_AirHoliday.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_AH = sum_yhat_AH + forecast_AirHoliday.iloc[y, 21]  # sum all the yhat  
ave_sum_y_AH = sum_y_AH / 70  # average of y  
ave_sum_yhat_AH = sum_yhat_AH / 70  # average of yhat  
acc_AH = 100 - ((ave_sum_y_AH - ave_sum_yhat_AH) / ave_sum_y_AH * 100)  # calculate the accuracy  
print("Accuracy_AH: " + str(acc_AH))  # print accuracy  
  
# Create a new forecast dataframe and display table  
forecast_AirBusiness = m_AB.predict(future_AirBusiness)  
print(forecast_AirBusiness[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_AirBusiness.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_AirBusiness.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_AirBusiness.yhat_upper.sum()))  
sum_y_AB = 0  
sum_yhat_AB = 0  
for x in range(0, 71):  
     sum_y_AB = sum_y_AB + df_AirBusiness.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_AB = sum_yhat_AB + forecast_AirBusiness.iloc[y, 21]  # sum all the yhat  
ave_sum_y_AB = sum_y_AB / 70  # average of y  
ave_sum_yhat_AB = sum_yhat_AB / 70  # average of yhat  
acc_AB = 100 - ((ave_sum_y_AB - ave_sum_yhat_AB) / ave_sum_y_AB * 100)  # calculate the accuracy  
print("Accuracy_AB: " + str(acc_AB))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_AirVFR = m_AV.predict(future_AirVFR)  
print(forecast_AirVFR[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_AirVFR.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_AirVFR.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_AirVFR.yhat_upper.sum()))  
sum_y_AV = 0  
sum_yhat_AV = 0  
for x in range(0, 71):  
     sum_y_AV = sum_y_AV + df_AirVFR.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_AV = sum_yhat_AV + forecast_AirVFR.iloc[y, 21]  # sum all the yhat  
ave_sum_y_AV = sum_y_AV / 70  # average of y  
ave_sum_yhat_AV = sum_yhat_AV / 70  # average of yhat  
acc_AV = 100 - ((ave_sum_yhat_AV - ave_sum_y_AV) / ave_sum_y_AV * 100)  # calculate the accuracy  
print("Accuracy_AV: " + str(acc_AV))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_AirStudy = m_AS.predict(future_AirStudy)  
print(forecast_AirStudy[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_AirStudy.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_AirStudy.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_AirStudy.yhat_upper.sum()))  
sum_y_AS = 0  
sum_yhat_AS = 0  
for x in range(0, 71):  
     sum_y_AS = sum_y_AS + df_AirStudy.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_AS = sum_yhat_AS + forecast_AirStudy.iloc[y, 21]  # sum all the yhat  
ave_sum_y_AS = sum_y_AS / 70  # average of y  
ave_sum_yhat_AS = sum_yhat_AS / 70  # average of yhat  
acc_AS = 100 - ((ave_sum_yhat_AS - ave_sum_y_AS) / ave_sum_y_AS * 100)  # calculate the accuracy  
print("Accuracy_AS: " + str(acc_AS))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_AirMiscellaneous = m_AM.predict(future_AirMiscellaneous)  
print(forecast_AirMiscellaneous[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_AirMiscellaneous.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_AirMiscellaneous.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_AirMiscellaneous.yhat_upper.sum()))  
sum_y_AM = 0  
sum_yhat_AM = 0  
for x in range(0, 71):  
     sum_y_AM = sum_y_AM + df_AirMiscellaneous.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_AM = sum_yhat_AM + forecast_AirMiscellaneous.iloc[y, 21]  # sum all the yhat  
ave_sum_y_AM = sum_y_AM / 70  # average of y  
ave_sum_yhat_AM = sum_yhat_AM / 70  # average of yhat  
acc_AM = 100 - ((ave_sum_y_AM - ave_sum_yhat_AM) / ave_sum_y_AM * 100)  # calculate the accuracy  
print("Accuracy_AM: " + str(acc_AM))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_SeaHoliday = m_SH.predict(future_SeaHoliday)  
print(forecast_SeaHoliday[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_SeaHoliday.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_SeaHoliday.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_SeaHoliday.yhat_upper.sum()))  
sum_y_SH = 0  
sum_yhat_SH = 0  
for x in range(0, 71):  
     sum_y_SH = sum_y_SH + df_SeaHoliday.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_SH = sum_yhat_SH + forecast_SeaHoliday.iloc[y, 21]  # sum all the yhat  
ave_sum_y_SH = sum_y_SH / 70  # average of y  
ave_sum_yhat_SH = sum_yhat_SH / 70  # average of yhat  
acc_SH = 100 - ((ave_sum_y_SH - ave_sum_yhat_SH) / ave_sum_y_SH * 100)  # calculate the accuracy  
print("Accuracy_SH: " + str(acc_SH))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_SeaBusiness = m_SB.predict(future_SeaBusiness)  
print(forecast_SeaBusiness[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_SeaBusiness.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_SeaBusiness.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_SeaBusiness.yhat_upper.sum()))  
sum_y_SB = 0  
sum_yhat_SB = 0  
for x in range(0, 71):  
     sum_y_SB = sum_y_SB + df_SeaBusiness.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_SB = sum_yhat_SB + forecast_SeaBusiness.iloc[y, 21]  # sum all the yhat  
ave_sum_y_SB = sum_y_SB / 70  # average of y  
ave_sum_yhat_SB = sum_yhat_SB / 70  # average of yhat  
acc_SB = 100 - ((ave_sum_yhat_SB - ave_sum_y_SB) / ave_sum_y_SB * 100)  # calculate the accuracy  
print("Accuracy_SB: " + str(acc_SB))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_SeaVFR = m_SV.predict(future_SeaVFR)  
print(forecast_SeaVFR[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_SeaVFR.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_SeaVFR.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_SeaVFR.yhat_upper.sum()))  
sum_y_SV = 0  
sum_yhat_SV = 0  
for x in range(0, 71):  
     sum_y_SV = sum_y_SV + df_SeaVFR.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_SV = sum_yhat_SV + forecast_SeaVFR.iloc[y, 21]  # sum all the yhat  
ave_sum_y_SV = sum_y_SV / 70  # average of y  
ave_sum_yhat_SV = sum_yhat_SV / 70  # average of yhat  
acc_SV = 100- ((ave_sum_yhat_SV - ave_sum_y_SV) / ave_sum_y_SV * 100)  # calculate the accuracy  
print("Accuracy_SV: " + str(acc_SV))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_SeaStudy = m_SS.predict(future_SeaStudy)  
print(forecast_SeaStudy[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_SeaStudy.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_SeaStudy.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_SeaStudy.yhat_upper.sum()))  
sum_y_SS = 0  
sum_yhat_SS = 0  
for x in range(0, 71):  
     sum_y_SS = sum_y_SS + df_SeaStudy.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_SS = sum_yhat_SS + forecast_SeaStudy.iloc[y, 21]  # sum all the yhat  
ave_sum_y_SS = sum_y_SS / 70  # average of y  
ave_sum_yhat_SS = sum_yhat_SS / 70  # average of yhat  
acc_SS = 100 - ((ave_sum_y_SS - ave_sum_yhat_SS) / ave_sum_y_SS * 100)  # calculate the accuracy  
print("Accuracy_SS: " + str(acc_SS))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_SeaMiscellaneous = m_SM.predict(future_SeaMiscellaneous)  
print(forecast_SeaMiscellaneous[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_SeaMiscellaneous.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_SeaMiscellaneous.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_SeaMiscellaneous.yhat_upper.sum()))  
sum_y_SM = 0  
sum_yhat_SM = 0  
for x in range(0, 71):  
     sum_y_SM = sum_y_SM + df_SeaMiscellaneous.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_SM = sum_yhat_SM + forecast_SeaMiscellaneous.iloc[y, 21]  # sum all the yhat  
ave_sum_y_SM = sum_y_SM / 70  # average of y  
ave_sum_yhat_SM = sum_yhat_SM / 70  # average of yhat  
acc_SM = 100 - ((ave_sum_yhat_SM - ave_sum_y_SM) / ave_sum_y_SM * 100)  # calculate the accuracy  
print("Accuracy_SM: " + str(acc_SM))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_TunnelHoliday = m_TH.predict(future_TunnelHoliday)  
print(forecast_TunnelHoliday[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_TunnelHoliday.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_TunnelHoliday.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_TunnelHoliday.yhat_upper.sum()))  
sum_y_TH = 0  
sum_yhat_TH = 0  
for x in range(0, 71):  
     sum_y_TH = sum_y_TH + df_TunnelHoliday.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_TH = sum_yhat_TH + forecast_TunnelHoliday.iloc[y, 21]  # sum all the yhat  
ave_sum_y_TH = sum_y_TH / 70  # average of y  
ave_sum_yhat_TH = sum_yhat_TH / 70  # average of yhat  
acc_TH = 100 - ((ave_sum_y_TH - ave_sum_yhat_TH) / ave_sum_y_TH * 100)  # calculate the accuracy  
print("Accuracy_TH: " + str(acc_TH))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_TunnelBusiness = m_TB.predict(future_TunnelBusiness)  
print(forecast_TunnelBusiness[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_TunnelBusiness.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_TunnelBusiness.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_TunnelBusiness.yhat_upper.sum()))  
sum_y_TB = 0  
sum_yhat_TB = 0  
for x in range(0, 71):  
     sum_y_TB = sum_y_TB + df_TunnelBusiness.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_TB = sum_yhat_TB + forecast_TunnelBusiness.iloc[y, 21]  # sum all the yhat  
ave_sum_y_TB = sum_y_TB / 70  # average of y  
ave_sum_yhat_TB = sum_yhat_TB / 70  # average of yhat  
acc_TB = 100 - ((ave_sum_y_TB - ave_sum_yhat_TB) / ave_sum_y_TB * 100)  # calculate the accuracy  
print("Accuracy_TB: " + str(acc_TB))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_TunnelVFR = m_TV.predict(future_TunnelVFR)  
print(forecast_TunnelVFR[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_TunnelVFR.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_TunnelVFR.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_TunnelVFR.yhat_upper.sum()))  
sum_y_TV = 0  
sum_yhat_TV = 0  
for x in range(0, 71):  
     sum_y_TV = sum_y_TV + df_TunnelVFR.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_TV = sum_yhat_TV + forecast_TunnelVFR.iloc[y, 21]  # sum all the yhat  
ave_sum_y_TV = sum_y_TV / 70  # average of y  
ave_sum_yhat_TV = sum_yhat_TV / 70  # average of yhat  
acc_TV = 100 - ((ave_sum_y_TV - ave_sum_yhat_TV) / ave_sum_y_TV * 100)  # calculate the accuracy  
print("Accuracy_TV: " + str(acc_TV))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_TunnelStudy = m_TS.predict(future_TunnelStudy)  
print(forecast_TunnelStudy[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_TunnelStudy.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_TunnelStudy.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_TunnelStudy.yhat_upper.sum()))  
sum_y_TS = 0  
sum_yhat_TS = 0  
for x in range(0, 71):  
     sum_y_TS = sum_y_TS + df_TunnelStudy.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_TS = sum_yhat_TS + forecast_TunnelStudy.iloc[y, 21]  # sum all the yhat  
ave_sum_y_TS = sum_y_TS / 70  # average of y  
ave_sum_yhat_TS = sum_yhat_TS / 70  # average of yhat  
acc_TS = 100 - ((ave_sum_yhat_TS - ave_sum_y_TS) / ave_sum_y_TS * 100)  # calculate the accuracy  
print("Accuracy_TS: " + str(acc_TS))  # print accuracy  
  
# # Create a new forecast dataframe and display table  
forecast_TunnelMiscellaneous = m_TM.predict(future_TunnelMiscellaneous)  
print(forecast_TunnelMiscellaneous[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())  
print("Forecasting value", int(forecast_TunnelMiscellaneous.yhat.sum()),  
     "; Forecasting lower bound", int(forecast_TunnelMiscellaneous.yhat_lower.sum()),  
     "; Forecasting upper bound", int(forecast_TunnelMiscellaneous.yhat_upper.sum()))  
sum_y_TM = 0  
sum_yhat_TM = 0  
for x in range(0, 71):  
     sum_y_TM = sum_y_TM + df_TunnelMiscellaneous.iloc[x, 1]  # sum all the y  
for y in range(0, 71):  
     sum_yhat_TM = sum_yhat_TM + forecast_TunnelMiscellaneous.iloc[y, 21]  # sum all the yhat  
ave_sum_y_TM = sum_y_TM / 70  # average of y  
ave_sum_yhat_TM = sum_yhat_TM / 70  # average of yhat  
acc_TM = 100 - ((ave_sum_y_TM - ave_sum_yhat_TM) / ave_sum_y_TM * 100)  # calculate the accuracy  
print("Accuracy_TM: " + str(acc_TM))  # print accuracy  
  
# Print the plot  
m_AH.plot(forecast_AirHoliday, xlabel='Year (Air & Holiday)', ylabel='Spend in million pound (Air & Holiday)')  
m_AB.plot(forecast_AirBusiness, xlabel='Year (Air & Business)', ylabel='Spend in million pound (Air & Business)')  
m_AV.plot(forecast_AirVFR, xlabel='Year (Air & VFR)', ylabel='Spend in million pound (Air & VFR)')  
m_AS.plot(forecast_AirStudy, xlabel='Year (Air & Study)', ylabel='Spend in million pound (Air & Study)')  
m_AM.plot(forecast_AirMiscellaneous, xlabel='Year (Air & Miscellaneous)', ylabel='Spend in million pound (Air & Miscellaneous)')  
  
m_SH.plot(forecast_SeaHoliday, xlabel='Year (Sea & Holiday)', ylabel='Spend in million pound (Sea & Holiday)')  
m_SB.plot(forecast_SeaBusiness, xlabel='Year (Sea & Business)', ylabel='Spend in million pound (Sea & Business)')  
m_SV.plot(forecast_SeaVFR, xlabel='Year (Sea & VFR)', ylabel='Spend in million pound (Sea & VFR)')  
m_SS.plot(forecast_SeaStudy, xlabel='Year (Sea & Study)', ylabel='Spend in million pound (Sea & Study)')  
m_SM.plot(forecast_SeaMiscellaneous, xlabel='Year (Sea & Miscellaneous)', ylabel='Spend in million pound (Sea & Miscellaneous)')  
  
m_TH.plot(forecast_TunnelHoliday, xlabel='Year (Tunnel & Holiday)', ylabel='Spend in million pound (Tunnel & Holiday)')  
m_TB.plot(forecast_TunnelBusiness, xlabel='Year (Tunnel & Business)', ylabel='Spend in million pound (Tunnel & Business)')  
m_TV.plot(forecast_TunnelVFR, xlabel='Year (Tunnel & VFR)', ylabel='Spend in million pound (Tunnel & VFR)')  
m_TS.plot(forecast_TunnelStudy, xlabel='Year (Tunnel & Study)', ylabel='Spend in million pound (Tunnel & Study)')  
m_TM.plot(forecast_TunnelMiscellaneous, xlabel='Year (Tunnel & Miscellaneous)', ylabel='Spend in million pound (Tunnel & Miscellaneous)')  
  
# Prophet checking method  
fcst_AH = forecast_AirHoliday[['ds', 'yhat']]  
fcst_AH = fcst_AH[fcst_AH['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_AirHoliday['y'], label='Train')  
plt.plot(test_df_AirHoliday['y'], label='Test')  
plt.plot(fcst_AH['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_AB = forecast_AirBusiness[['ds', 'yhat']]  
fcst_AB = fcst_AB[fcst_AB['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_AirBusiness['y'], label='Train')  
plt.plot(test_df_AirBusiness['y'], label='Test')  
plt.plot(fcst_AB['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_AV = forecast_AirVFR[['ds', 'yhat']]  
fcst_AV = fcst_AV[fcst_AV['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_AirVFR['y'], label='Train')  
plt.plot(test_df_AirVFR['y'], label='Test')  
plt.plot(fcst_AV['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_AS = forecast_AirStudy[['ds', 'yhat']]  
fcst_AS = fcst_AS[fcst_AS['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_AirStudy['y'], label='Train')  
plt.plot(test_df_AirStudy['y'], label='Test')  
plt.plot(fcst_AS['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_AM = forecast_AirMiscellaneous[['ds', 'yhat']]  
fcst_AM = fcst_AM[fcst_AM['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_AirMiscellaneous['y'], label='Train')  
plt.plot(test_df_AirMiscellaneous['y'], label='Test')  
plt.plot(fcst_AM['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_SH = forecast_SeaHoliday[['ds', 'yhat']]  
fcst_SH = fcst_SH[fcst_SH['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_SeaHoliday['y'], label='Train')  
plt.plot(test_df_SeaHoliday['y'], label='Test')  
plt.plot(fcst_SH['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_SB = forecast_SeaBusiness[['ds', 'yhat']]  
fcst_SB = fcst_SB[fcst_SB['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_SeaBusiness['y'], label='Train')  
plt.plot(test_df_SeaBusiness['y'], label='Test')  
plt.plot(fcst_SB['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_SV = forecast_SeaVFR[['ds', 'yhat']]  
fcst_SV = fcst_SV[fcst_SV['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_SeaVFR['y'], label='Train')  
plt.plot(test_df_SeaVFR['y'], label='Test')  
plt.plot(fcst_SV['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_SS = forecast_SeaStudy[['ds', 'yhat']]  
fcst_SS = fcst_SS[fcst_SS['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_SeaStudy['y'], label='Train')  
plt.plot(test_df_SeaStudy['y'], label='Test')  
plt.plot(fcst_SS['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_SM = forecast_SeaMiscellaneous[['ds', 'yhat']]  
fcst_SM = fcst_SM[fcst_SM['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_SeaMiscellaneous['y'], label='Train')  
plt.plot(test_df_SeaMiscellaneous['y'], label='Test')  
plt.plot(fcst_SM['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_TH = forecast_TunnelHoliday[['ds', 'yhat']]  
fcst_TH = fcst_TH[fcst_TH['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_TunnelHoliday['y'], label='Train')  
plt.plot(test_df_TunnelHoliday['y'], label='Test')  
plt.plot(fcst_TH['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_TB = forecast_TunnelBusiness[['ds', 'yhat']]  
fcst_TB = fcst_TB[fcst_TB['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_TunnelBusiness['y'], label='Train')  
plt.plot(test_df_TunnelBusiness['y'], label='Test')  
plt.plot(fcst_TB['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_TV = forecast_TunnelVFR[['ds', 'yhat']]  
fcst_TV = fcst_TV[fcst_TV['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_TunnelVFR['y'], label='Train')  
plt.plot(test_df_TunnelVFR['y'], label='Test')  
plt.plot(fcst_TV['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_TS = forecast_TunnelStudy[['ds', 'yhat']]  
fcst_TS = fcst_TS[fcst_TS['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_TunnelStudy['y'], label='Train')  
plt.plot(test_df_TunnelStudy['y'], label='Test')  
plt.plot(fcst_TS['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
# Prophet checking method  
fcst_TM = forecast_TunnelMiscellaneous[['ds', 'yhat']]  
fcst_TM = fcst_TM[fcst_TM['ds'] >= '2018-12-31']  
plt.figure(figsize=(10,6))  
plt.plot(train_df_TunnelMiscellaneous['y'], label='Train')  
plt.plot(test_df_TunnelMiscellaneous['y'], label='Test')  
plt.plot(fcst_TM['yhat'], label='Prophet Forecast')  
plt.legend(loc='best')  
  
plt.show()  
