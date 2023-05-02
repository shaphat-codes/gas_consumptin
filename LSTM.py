import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout, Dense
from sklearn.metrics import mean_squared_error



df = pd.read_csv("final_gas_report.csv")

df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

train = df.loc[df.index >= '2020-01-01']
test = df.loc[df.index >= '2022-09-30']



train = df.loc[df.index >= '2020-01-01']
test = df.loc[df.index >= '2022-09-30']



def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    
    
    
    
    return df

df = create_features(df)



train = create_features(train)
test = create_features(test)

FEATURES = [ 'dayofweek', 'quarter', 'month',  'holiday']
TARGET = 'total'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

regressor = Sequential()


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting 
regressor.fit(X_train, y_train, epochs = 100, batch_size = 10)

#making prediction using our test data
test['prediction'] = regressor.predict(X_test)
print(test['prediction'])

#calculating and printing our RMSE
score = np.sqrt(mean_squared_error(test['total'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')




