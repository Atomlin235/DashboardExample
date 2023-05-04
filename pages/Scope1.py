import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Scope 1 Estimation", page_icon="📈")

st.markdown("# Estimating scope 1 Emmissions")
st.sidebar.header("Scope 1 Estimation")

# Setting Up Linear Regression Model

client_measure_df = pd.read_csv('Client Measurement Data Fix v1.0.csv')

client_measure_df.drop(client_measure_df[client_measure_df['Scope 1 total KGCO2e'] == 0].index, inplace = True)


client_measure_df= client_measure_df.replace('-',np.nan)
client_measure_df= client_measure_df.replace('',np.nan)
client_measure_df= client_measure_df.replace(' ',np.nan)
client_measure_df = client_measure_df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
client_measure_df['# Sites'] = client_measure_df['# Sites'].fillna(1)
client_measure_df['# Employees'] = client_measure_df['# Employees'].fillna(0)
client_measure_df['Measurement No.'] = client_measure_df['Measurement No.'].fillna(1)

pd.set_option('display.float_format', '{:.2f}'.format)

drop_scope_df =  client_measure_df.dropna(subset=['Scope 1 total KGCO2e'])
new_df = client_measure_df.dropna(subset=['Stationary Combustion'])


drop_empty_df = new_df.dropna(subset=['Sector / Industry','Measurement No.','Client Name','Reporting Year End Date','# Employees'])
drop_lease_franchise_processdf = drop_empty_df.drop(columns=['Leased Assets', 'Leased Assets .','Franchises','Processing of Sold Products','Use of Sold Products','Investments'])

clean_df=drop_lease_franchise_processdf.dropna(subset=['Purchased Electricity'])


statCom = clean_df['Stationary Combustion']
scope1 = clean_df['Scope 1 total KGCO2e']

statCom_np = statCom.to_numpy()
scope1_np = scope1.to_numpy()

sklearn_model = LinearRegression().fit(statCom_np.reshape((109,1)), scope1_np)
sklearn_scope1_predictions = sklearn_model.predict(statCom_np.reshape((109, 1)))


number = st.number_input('Insert Stationary Combustion Value ')
st.write('The current number Staionary combustion is ', number)

first_row = np.array([number])

scope1_user_pred = sklearn_model.predict(first_row.reshape(1, -1))
st.write('The estimated scope 1 Value is ', scope1_user_pred.flat[0] )


#Predictions Data frame 
cliemtname = clean_df['Client Name']
statCom_s = clean_df['Stationary Combustion']
scope1_s =  clean_df['Scope 1 total KGCO2e']
predictions_df = pd.DataFrame({'Client Name':cliemtname,'Stationary Combustion':statCom_s,
                               'Scope 1 total KGCO2e':scope1_s,
                               'Sklearn scope 1 Predictions':sklearn_scope1_predictions})
predictions_df['change'] = (predictions_df['Sklearn scope 1 Predictions'] - predictions_df['Scope 1 total KGCO2e'])/ predictions_df['Scope 1 total KGCO2e'] * 100
predictions_df.round(4)

y_true = predictions_df['Scope 1 total KGCO2e'].values.tolist()
y_pred = predictions_df['Sklearn scope 1 Predictions'].values.tolist()

st.write("The mean absolute error for data used is: " ,mean_absolute_error(y_true, y_pred))
st.write("The mean absolute error for data used percentage is: " ,mean_absolute_percentage_error(y_true, y_pred))



# Predictions data frame to be inserted below estimation 

with st.expander("See Model Data"):
    st.dataframe(predictions_df)


st.markdown("### Explanation of Scope 1 results")

st.write(
    """ A summary of the linear regression model for estimating scope 1 emmissions using stationary combustion.

The accuracy of predicting scope 1 emmisions based on stationary combustion is too low to be considered to be used. This low accuracy may be potentially linked to the volume of missing and 0 value data present in the columns that produce the total for scope 1 emmissions. In order generate a more accurate model more data is needed that does not consist of 0 values. The results of accuracy for the model are explained below.

Mean absolute Error: 101053.87784241795

In the context of machine learning, absolute error refers to the magnitude of difference between the prediction of an observation and the true value of that observation. MAE takes the average of absolute errors for a group of predictions and observations as a measurement of the magnitude of errors for the entire group. MAE can also be referred as L1 loss function. As can be ssen the mean absolute error for this model is considerably high


Mean Absolute percentage Error: 40.67376511434855

Mean absolute percentage error measures the average magnitude of error produced by a model, or how far off predictions are on average. A MAPE value of 40% means that the average absolute percentage difference between the predictions and the actuals is 40%. In other words, the model’s predictions are, on average, off by 40% from the real values. Suggesting that more data is required to make the model to be accurately used.This would require further attention / development in future models.
 """
)