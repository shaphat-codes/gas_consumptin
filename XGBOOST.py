import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


#loading the data and defining our index
df = pd.read_csv('final_gas_report.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)


#allocating data for testing and training.
train = df.loc[df.index >= '2020-01-01']
test = df.loc[df.index >= '2022-09-30']


#creating features from dataset
def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)

#defingin features and targets for traingin and testing data
train = create_features(train)
test = create_features(test)

FEATURES = [ 'dayofweek', 'quarter', 'month',  'holiday']
TARGET = 'total'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


#fitting using the XGBOOST regressor
reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.9)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=1)




#making prediction from our test data
test['prediction'] = reg.predict(X_test)

#printing some of the predictinos made
print(test['prediction'])

#calculating and printing the root mean squared error of the model
score = np.sqrt(mean_squared_error(test['total'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')

