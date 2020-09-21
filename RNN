# Set validation   
from keras.models import Sequential  
from keras.layers import Dense, Activation, Dropout  
from keras.callbacks import EarlyStopping  
from keras.optimizers import adam  
  
# Add series of layers to create the network. The first layer needs input_shape information.  
# Build the neural network   
model=Sequential()  
model.add(Dense(25,input_dim=x.shape[1], activation='relu')) #Hidden 1  
model.add(Dropout(0.4))  
model.add(Dense(10,activation='relu')) #Hidden 2  
model.add(Dropout(0.4))  
model.add(Dense(1)) #Output  
  
# The compilation step: configure our model parameters for training  
model.compile(loss='mean_squared_error',optimizer=adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,amsgrad=False))  
  
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5,verbose=1,mode='auto',restore_best_weights=True)  
# The fit() function returns the training history/logs that we could use later to analyse training, validation lossess and accuracy  
history=model.fit(x_train[:11961],y_train[:11961],validation_data=(x_val,y_val),callbacks=[monitor],verbose=2,epochs=1000)  
  
from sklearn import metrics  
# Build the prediction list and calculate the error.  
pred=model.predict(x_val)  
testdata=model.predict(x_test)  
#measure MSE error  
score=metrics.mean_squared_error(pred,y_val)  
print("Validation score (MSE): {}". format(score))  
score=metrics.mean_squared_error(testdata,y_test)  
print("Test score (MSE): {} ".format(score))  
  
5-fold cross validation   
from sklearn.model_selection import train_test_split  
import pandas as pd  
import os  
import numpy as np  
from sklearn import metrics  
from scipy.stats import zscore  
from sklearn.model_selection import KFold  
from keras.callbacks import EarlyStopping  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Activation, Dropout  
  
# Dividing the dataset into train data and test data with 0.8, 0.2, holdout using for testing   
x_main, x_holdout, y_main, y_holdout = train_test_split(x, y, test_size=0.20)   
  
# Cross-validate  
kf = KFold(5)  
      
oos_y = []  
oos_pred = []  
fold = 0  
for train, test in kf.split(x_main):          
    fold+=1  
    print(f"Fold #{fold}")  
          
    x_train = x_main[train]  
    y_train = y_main[train]  
    x_test = x_main[test]  
    y_test = y_main[test]  
  # Build the neural network, we used same neural network, but without using early stopping  
    model = Sequential()  
    model.add(Dense(25, input_dim=x.shape[1], activation='relu')) #Hidden 1  
    model.add(Dropout(0.2))  
    model.add(Dense(10, activation='relu')) #Hidden 2  
    model.add(Dropout(0.2))  
    model.add(Dense(1)) #Output  
  
    model.compile(loss='mean_squared_error', optimizer='adam')  
    history=model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=0,epochs=50)  
      
    pred = model.predict(x_test)  
      
    oos_y.append(y_test)  
    oos_pred.append(pred)   
  
    # Measure accuracy  
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))  
    print(f"Fold score (RMSE): {score}")  
  
# Build the oos prediction list and calculate the error.  
oos_y = np.concatenate(oos_y)  
oos_pred = np.concatenate(oos_pred)  
score = np.sqrt(metrics.mean_squared_error(oos_pred,oos_y))  
print()  
print(f"Cross-validated score (RMSE): {score}")      
      
# Write the cross-validated prediction (from the last neural network)  
holdout_pred = model.predict(x_holdout)  
  
score = np.sqrt(metrics.mean_squared_error(holdout_pred,y_holdout))  
print(f"Holdout score (RMSE): {score}")   
  
# For both validation techniques  
# Let us visualize our neural network architecture  
model.summary()  
  
#Plotting the testing and validation errors, which return from a fit() function   
import matplotlib.pyplot as plt  
  
history_dict= history.history  
print(history_dict)  
  
train_loss=history_dict['loss']  
val_loss=history_dict['val_loss']  
  
plt.plot(train_loss,'bo-',label='Train loss')  
plt.plot(val_loss,'ro-', label='Val loss')  
  
plt.title('Training and validation loss')  
plt.xlabel('Epochs')  
plt.ylabel('Loss')  
plt.title('train loss and validation loss')  
plt.legend()  
plt.show()  
  
#Plotting function to plot lift chart  
def chart_regression(pred,y, sort= True):  
  t=pd.DataFrame({'pred':pred, 'y': y.flatten()})  
  if sort:  
    t.sort_values(by=['y'],inplace=True)  
  plt.plot(t['y'].tolist(), label='expected')  
  plt.plot(t['pred'].tolist(), label='predication')  
  plt.ylabel('output')  
  plt.xlabel('Sample Number')  
  plt.title("After testing the model validation set approch")  
  plt.legend()  
  plt.show  
  
#Plot the chart  
chart_regression(testdata.flatten(),y_test)  

def reverse_zscore(pandas_series, mean, std):
'''Mean and standard deviation should be of original variable before standardization'''

yis=pandas_series*std+mean
   return yis
 
original_mean, original_std = mean_std_spend
original_var_series = reverse_zscore(testdata, original_mean, original_std)
