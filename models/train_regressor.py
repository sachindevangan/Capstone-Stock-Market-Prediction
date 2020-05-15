# importing libraries
import warnings
warnings.filterwarnings("ignore")
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
from sqlalchemy import create_engine
import pickle
from sklearn.metrics import r2_score,mean_absolute_error

train_period = 7
predict_period = 1
n_day_later_predict= 7

def get_rolling_data(X,y,train_period,predict_period=1,n_day_later_predict=1):
    """ 
    Generating Timeseries Input And Output Data.  
  
    Parameters: 
    X,y (DataFrame): Features,Labels
    train_period (int): Timesteps For Model
    predict_period (int): Predict On The nth Day Of The End Of The Training Window
    
    Returns: 
    rolling_X (DataFrame): Features
    rolling_y (DataFrame): Labels
    """
    assert X.shape[0] == y.shape[0], (
            'X.shape: %s y.shape: %s' % (X.shape, y.shape))
    
    rolling_X, rolling_y = [],[]
    
    for i in range(len(X)-train_period-predict_period-(n_day_later_predict)):

        curr_X=X.iloc[i:i+train_period,:]
        curr_y=y.iloc[i+train_period+n_day_later_predict:i+train_period+predict_period+n_day_later_predict]
        rolling_X.append(curr_X.values.tolist())
        if predict_period == 1:
            rolling_y.append(curr_y.values.tolist()[0])
        else:
            rolling_y.append(curr_y.values.tolist())
        
    rolling_X = np.array(rolling_X)
    rolling_y = np.array(rolling_y)
    return rolling_X, rolling_y


def load_data(database_filepath):
    """ 
    Loading Data From Database. 
  
    Splitting X And Y Columns As TimeSeries Data By Calling get_rolling_data Method. 
  
    Parameters: 
    database_filepath (str): Filepath Where Database Is Located.
    
    Returns: 
    X (DataFrame): Features
    Y (DataFrame): Labels
    """
    
    # loading data from database
    db_name = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(db_name)

    # using pandas to read table from database
    df = pd.read_sql_table('Stock',engine)

    rolling_X, rolling_y = get_rolling_data(df, df.loc[:,'Stock_Adj Close'], train_period=train_period, 
                                        predict_period=predict_period,
                                        n_day_later_predict=n_day_later_predict)

    return rolling_X , rolling_y




class ModelData():
    '''Data Class For Model Train, Predict And Validate.'''
    def __init__(self,X,y,seed=None,shuffle=True):
        
        self._seed = seed 
        np.random.seed(self._seed)
        
        assert X.shape[0] == y.shape[0], (
            'X.shape: %s y.shape: %s' % (X.shape, y.shape))
        self._num_examples = X.shape[0]
        
        # If shuffle
        if shuffle:
            np.random.seed(self._seed)
            randomList = np.arange(X.shape[0])
            np.random.shuffle(randomList)
            self._X, self._y = X[randomList], y[randomList] 
        
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    
    def train_validate_test_split(self,validate_size=0.10,test_size=0.10):
        '''Train, Predict And Validate Splitting Function'''
        validate_start = int(self._num_examples*(1-validate_size-test_size)) + 1
        test_start = int(self._num_examples*(1-test_size)) + 1
        if validate_start > len(self._X) or test_start > len(self._X):
            pass
        train_X,train_y = self._X[:validate_start],self._y[:validate_start]
        validate_X, validate_y = self._X[validate_start:test_start],self._y[validate_start:test_start]
        test_X,test_y = self._X[test_start:],self._y[test_start:]
        
        if test_size == 0:
            return ModelData(train_X,train_y,self._seed), ModelData(validate_X,validate_y,self._seed)
        else:
            return ModelData(train_X,train_y,self._seed), ModelData(validate_X,validate_y,self._seed), ModelData(test_X,test_y,self._seed)

    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y

def build_model():
    """
    Build Model Function.
    
    This Function's Output Is A Dictionary Of 3 Best Regressor Models i.e. XGB Regressor,
    Catboost Regressor And LGBM Regressor.

    Returns: 
    model (Dict) : A Dictionary Of Regressor Models
    """
    # xgb regressor
    xgb_reg = xgb.XGBRegressor(n_estimators=10000,min_child_weight= 40,learning_rate=0.01,colsample_bytree = 1,subsample = 0.9)
    # catboost regressor
    cat_reg = CatBoostRegressor(iterations=10000,learning_rate=0.005,loss_function = 'RMSE')
    # lgbm regressor
    lgbm_reg = lgb.LGBMRegressor(num_leaves=31,learning_rate=0.001, max_bin = 30,n_estimators=10000)
    
    model = {'xgb':xgb_reg,'cat':cat_reg,'lgbm':lgbm_reg}                    
                       
                        

    return model


def evaluate_model(model, X_test, Y_test):
    """ 
    Model Evaluation Function. 
  
    Evaluating The Models On Test Set And Computing R2 Score And Mean Absolute Error. 
  
    Parameters:
    model (Dict) : A Dictionary Of Trained Regressor Models
    X_test (DataFrame) : Test Features
    Y_test (DataFrame) : Test Labels
    
    """

    # predict on test data
    pred = (model['xgb'].predict(X_test) + model['cat'].predict(X_test) + model['lgbm'].predict(X_test)) / 3
    # rescaling the predictions
    real = np.exp(Y_test)
    pred = np.exp(pred)
    # computing the r2 score
    print('R2 Score :')
    print(r2_score(real,pred))
    # computing the mean absolute error
    print('Mean Absolute Error :')
    print(mean_absolute_error(real,pred))
    

def save_model(model, model_filepath):
    """
    Save Model function
    
    This Function Saves Trained Models As Pickle File, To Be Loaded Later.
    
    Parameters:
    model (Dict) : A Dictionary Of Trained Regressor Models
    model_filepath (str) : Destination Path To Save .pkl File
    
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        model_data = ModelData(X, Y,seed=666,shuffle=False)
        model_train_data, model_validate_data = model_data.train_validate_test_split(validate_size=0.10,test_size=0)
        y_train = model_train_data.y[:,np.newaxis]
        y_validate = model_validate_data.y[:,np.newaxis]

        X_train = model_train_data.X
        X_validate = model_validate_data.X
        
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_validate = X_validate.reshape((X_validate.shape[0],X_validate.shape[1]*X_validate.shape[2]))
		
        
        print('Building model...')
        model = build_model()
        print('Training XGB model...')
        model['xgb'].fit(X_train, y_train,eval_set = [(X_validate[:300],y_validate[:300])],early_stopping_rounds = 50,verbose = False)
        print('Training Catboost model...')
        model['cat'].fit(X_train, y_train,eval_set = [(X_validate[:300],y_validate[:300])],early_stopping_rounds = 50,verbose = False)
        print('Training Lgbm model...')
        model['lgbm'].fit(X_train, y_train,eval_set = [(X_validate[:300],y_validate[:300].ravel())],early_stopping_rounds = 50,verbose = False)

        print('Evaluating Combined model...')
        evaluate_model(model, X_validate, y_validate)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the Stock database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_regressor.py ../data/Stock.db regressor.pkl')


if __name__ == '__main__':
    main()
