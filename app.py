from flask import Flask, render_template, request
import covidcast
from datetime import date
from matplotlib import pyplot as plt
import pandas as pd

app = Flask(__name__)

def generate(): 
  possible_factors = [
                    ["safegraph", "part_time_work_prop"], 
                    ["fb-survey", "smoothed_cli"], 
                    ["fb-survey", "smoothed_hh_cmnty_cli"],
                    ["doctor-visits", "smoothed_adj_cli"],
                    ["indicator-combination", "confirmed_7dav_incidence_num"]
                    ] 
  x = []
  dt_temp = []
  for value in possible_factors: 
    source, signal = value
    temp = covidcast.signal(source, signal,
                         date(2020, 11, 1), date(2020, 11, 30),
                         geo_type="state")
    x.append(temp)
    dt_temp.append(3)

  x.append(covidcast.signal("indicator-combination", "confirmed_incidence_num",
                         date(2020, 11, 1), date(2020, 11, 30),
                         geo_type="state"))
  dt_temp.append(0)
  df = covidcast.aggregate_signals(x, dt = dt_temp)
  df.head()
  return df

possible_factors = ["cases", "Away from Home 3-6hr a Day", "COVID-Like Symptoms in Community", "COVID-Related Doctor Visits", "COVID-Like Symptoms"]
df = generate() 

df = df.rename(
    columns={"safegraph_part_time_work_prop_0_value": "Away from Home 3-6hr a Day",
             "fb-survey_smoothed_cli_1_value": "COVID-Like Symptoms",
             "fb-survey_smoothed_hh_cmnty_cli_2_value": "COVID-Like Symptoms in Community",
             "doctor-visits_smoothed_adj_cli_3_value": "COVID-Related Doctor Visits",
             "indicator-combination_confirmed_7dav_incidence_num_4_value": "cases",
             "indicator-combination_confirmed_incidence_num_5_value": "cases_future"})


df = df[["time_value", "geo_value", "Away from Home 3-6hr a Day", "COVID-Like Symptoms", "COVID-Like Symptoms in Community", "COVID-Related Doctor Visits", "cases", "cases_future"]]

df.dropna(inplace=True)


import sklearn # import scikit-learn
from sklearn import preprocessing # import preprocessing utilites
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import numpy as np

def run_linear_regression(myList):
  y = df["cases_future"]

  features_cat = []
  features_num = myList

  X_cat = df[features_cat]
  X_num = df[features_num]

  X_cat = X_cat.fillna(0)
  enc = preprocessing.OneHotEncoder()
  enc.fit(X_cat) # fit the encoder to categories in our data 
  one_hot = enc.transform(X_cat) # transform data into one hot encoded sparse array format
  
  # Finally, put the newly encoded sparse array back into a pandas dataframe so that we can use it
  X_cat_proc = pd.DataFrame(one_hot.toarray(), columns=enc.get_feature_names())
  X_cat_proc.head()

  scaled = preprocessing.scale(X_num)
  X_num_proc = pd.DataFrame(scaled, columns=features_num)

  X = pd.concat([X_num], axis=1, sort=False)
  X.head()
  X = X.fillna(0)

  X_train, X_TEMP, y_train, y_TEMP = train_test_split(X, y, test_size=0.20) # split out into training 70% of our data
  X_validation, X_test, y_validation, y_test = train_test_split(X_TEMP, y_TEMP, test_size=0.50) # split out into validation 15% of our data and test 15% of our data
  print(X_train.shape, X_validation.shape, X_test.shape) # print data shape to check the sizing is correct

  reg = LinearRegression().fit(X_train, y_train)
  y_pred = reg.predict(X_validation)

  #if (y_pred.shape != y_test.shape): y_test[:-1]
  #fig, ax = plt.subplots()
  #ax.scatter(y_pred, y_test[:-1], edgecolors=(0, 0, 1))
  #ax.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'r--', lw=3)
  #ax.set_xlabel('Predicted')
  #ax.set_ylabel('Actual')
  #plt.show()

  mae = metrics.mean_absolute_error(y_validation, y_pred)
  mse = metrics.mean_squared_error(y_validation, y_pred)
  r2 = metrics.r2_score(y_validation, y_pred)

  print("The model performance for testing set")
  print("--------------------------------------")
  print('MAE is {}'.format(mae))
  print('MSE is {}'.format(mse))
  print('R2 score is {}'.format(r2))
  print(myList)
  print('Coefficients: \n', reg.coef_)
  print("what", reg.get_params)

  return (list(y_validation), list(y_pred), format(mae), format(mse), format(r2))






@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      state = request.form['state']
      factors = []
      for factor in possible_factors:
      	factors.append(factor)

      y_val, y_pred, mae, mse, r2 = run_linear_regression(factors)
      return render_template("result.html",state = state, mae=mae, mse=mse, r2=r2)

if __name__ == '__main__':
   app.run(debug = True)