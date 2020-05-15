# Udacity DSND Capstone Investment And Trading Project


## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installing)
	3. [Executing Program](#executing)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Screenshots](#screenshots)

<a name="descripton"></a>
## Description

This Project is part of Data Science Nanodegree Program by Udacity.
The initial dataset contains stock price of COAL INDIA of past 15 years from Yahoo Finance. 
The aim of the project is to build a Stock Price Predictor Which Can predict price of stock 7 days in advance.

The Project is divided in the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper database structure
2. Machine Learning Pipeline to train a model able to predict price of stock 7 days in advance
3. Web App to show model results in real time. 

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
All Dependencies Are Specified In requirements.txt
Install Dependencies Following Way:
```
pip install -r requirements.txt
```

<a name="installing"></a>
### Installing
Clone this GIT repository:
```
git clone 
```
<a name="executing"></a>
### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/COALINDIA.NS.csv data/Stock.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_regressor.py data/Stock.db models/regressor.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/


<a name="authors"></a>
## Authors

* [Sachin Devangan](https://github.com/sachindevangan)

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program
* [Yahoo Finance](https://in.finance.yahoo.com/) for providing stock price dataset to train my model

<a name="screenshots"></a>
## Screenshots

1. This is an example of a date you can type and select the stock to test Machine Learning model performance

![Sample Input](screenshots/sample_input.png)

2. After clicking **Try it Now**, you can see the predicted price and the actual price if available.

![Sample Output](screenshots/sample_output.png)

3. The main page

![Main Page](screenshots/main_page.png)
