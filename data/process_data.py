import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def technical_indicators(df):
    """
    Technical Indicator Calculator Function.
    
    This Function's Output Is A Pandas DataFrame Of Various Techincal Indicators Such As RSI,SMA,EVM,EWMA
    BB And ROC Using Different Time Intervals.
    
    Parameters:
    df (DataFrame) : Pandas DataFrame Of Stock Price

    Returns: 
    new_df (DataFrame) : Pandas DataFrame Of Techincal Indicators
    """
    
    new_df = pd.DataFrame()
    
    dm = ((df['Stock_High'] + df['Stock_Low'])/2) - ((df['Stock_High'].shift(1) + df['Stock_Low'].shift(1))/2)
    br = (df['Stock_Volume'] / 100000000) / ((df['Stock_High'] - df['Stock_Low']))
    EVM = dm / br 
    new_df['EVM_15'] = EVM.rolling(15).mean()

    
    sma_60 = pd.Series.rolling(df['Stock_Close'], window=60, center=False).mean()
    new_df["SMA_60"] = sma_60
    
    sma_200 = pd.Series.rolling(df['Stock_Close'], window=30, center=False).mean()
    new_df["SMA_200"] = sma_200
    
    ewma_50 = df['Stock_Close'].ewm(span = 50, min_periods = 50 - 1).mean()
    new_df["EWMA_50"] = ewma_50
    
    ewma_200 = df['Stock_Close'].ewm(span = 200, min_periods = 200 - 1).mean()
    new_df["EWMA_200"] = ewma_200
    
    sma_5 = pd.Series.rolling(df['Stock_Close'], window=5, center=False).mean()
    std_5 = pd.Series.rolling(df['Stock_Close'], window=5, center=False).std()
    bb_5_upper = sma_5 + (2 * std_5)
    bb_5_lower = sma_5 - (2 * std_5)
    new_df["BB_5_UPPER"] = bb_5_upper
    new_df["BB_5_LOWER"] = bb_5_lower
    new_df["SMA_5"] = sma_5
    
    sma_10 = pd.Series.rolling(df['Stock_Close'], window=10, center=False).mean()
    std_10 = pd.Series.rolling(df['Stock_Close'], window=10, center=False).std()
    bb_10_upper = sma_10 + (2 * std_10)
    bb_10_lower = sma_10 - (2 * std_10)
    new_df["BB_10_UPPER"] = bb_10_upper
    new_df["BB_10_LOWER"] = bb_10_lower
    new_df["SMA_10"] = sma_10
    
    sma_20 = pd.Series.rolling(df['Stock_Close'], window=20, center=False).mean()
    std_20 = pd.Series.rolling(df['Stock_Close'], window=20, center=False).std()
    bb_20_upper = sma_20 + (2 * std_20)
    bb_20_lower = sma_20 - (2 * std_20)
    new_df["BB_20_UPPER"] = bb_20_upper
    new_df["BB_20_LOWER"] = bb_20_lower
    new_df["SMA_20"] = sma_20
    
    roc_5 = df['Stock_Close'][5:]/df['Stock_Close'][:-5].values - 1
    new_df["ROC_5"] = roc_5
    
    roc_10 = df['Stock_Close'][10:]/df['Stock_Close'][:-10].values - 1
    new_df["ROC_10"] = roc_10
    
    roc_20 = df['Stock_Close'][20:]/df['Stock_Close'][:-20].values - 1
    new_df["ROC_20"] = roc_20
    
    delta = df['Stock_Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    up_5 = pd.Series.rolling(up, window=5, center=False).mean()
    down_5 = pd.Series.rolling(down.abs(), window=5, center=False).mean()
    RS_5 = up_5 / down_5
    RSI_5 = 100.0 - (100.0 / (1.0 + RS_5))
    new_df["RSI_5"] = RSI_5
    
    up_10 = pd.Series.rolling(up, window=10, center=False).mean()
    down_10 = pd.Series.rolling(down.abs(), window=10, center=False).mean()
    RS_10 = up_10 / down_10
    RSI_10 = 100.0 - (100.0 / (1.0 + RS_10))
    new_df["RSI_10"] = RSI_10
    
    up_20 = pd.Series.rolling(up, window=20, center=False).mean()
    down_20 = pd.Series.rolling(down.abs(), window=20, center=False).mean()
    RS_20 = up_20 / down_20
    RSI_20 = 100.0 - (100.0 / (1.0 + RS_20))
    new_df["RSI_20"] = RSI_20
    
    return new_df


def process_data(df):
    """
    Data Pre-Processing And Cleaning Function.
    
    This Function's Output Is A Pandas DataFrame Which Has No Missing Values And
    And Prices Are Log Scaled Along With Various Technical Indicators.
    
    Parameters:
    df (DataFrame) : Pandas DataFrame Of Stock Price

    Returns: 
    df (DataFrame) : New Pandas DataFrame Of Scaled Features
    """


    df.columns = ['Date', 'Stock_Open', 'Stock_High', 'Stock_Low', 'Stock_Close', 'Stock_Adj Close', 'Stock_Volume']
    df = df.drop(['Date'],axis = 1)
    temp = df
    
    # Fill missing values in data frame, in place.
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
        
    # log
    epsilon = 1e-5
    df = df.apply(lambda x:np.log(x + epsilon),axis=1)

    temp = technical_indicators(temp)
    temp.fillna(method="ffill", inplace=True)
    temp.fillna(method="bfill", inplace=True)
    
    df = pd.concat([df,temp],axis = 1)
    
    
    return df


def save_data(df, database_filename):
    """ 
    Save To Database Method. 
  
    Saving The Cleaned Pandas DataFrame Into A SQLite Database. 
  
    Parameters: 
    df (DataFrame): Pandas DataFrame Object Containing The Cleaned Data.
    database_filename (str) : Filename Of Database.
  
    """

    # formatting database name properly
    db_name = 'sqlite:///{}'.format(database_filename)

    # initiating SQLAlchemy Engine
    engine = create_engine(db_name)

    # using pandas to save the DataFrame to the database
    df.to_sql('Stock', engine, index=False)
 


def main():
    if len(sys.argv) == 3:

        stock_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    STOCK: {}\n  '
              .format(stock_filepath))
        
        df = pd.read_csv(stock_filepath)

        print('Cleaning data and computing technical indicators...')
        df = process_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepath of the stock '\
              'datasets as the first  as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the second argument. \n\nExample: python process_data.py '\
              'data/ITC.NS.csv '\
              'data/Stock.db')


if __name__ == '__main__':
    main()
