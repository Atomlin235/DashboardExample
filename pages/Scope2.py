import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Scope 2 Estimation", page_icon="📈")

st.markdown("# Estimating scope 2 Emmissions")
st.sidebar.header("Scope 2 Estimation")


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

purchElec = clean_df['Purchased Electricity']
scope2 = clean_df['Scope 2 total KGCO2e']

purchElec_np = purchElec.to_numpy()
scope2_np = scope2.to_numpy()

sklearn_model = LinearRegression().fit(purchElec_np.reshape((109,1)), scope2_np)
sklearn_scope2_predictions = sklearn_model.predict(purchElec_np.reshape((109, 1)))

number = st.number_input('Insert Purchased Electricity Value ')
st.write('The current number Purchased Electricity is ', number)

first_row = np.array([number])

scope2_user_pred = sklearn_model.predict(first_row.reshape(1, -1))
st.write('The estimated scope 2 Value is ', scope2_user_pred.flat[0] )


cliemtname = clean_df['Client Name']
purchElec_s = clean_df['Purchased Electricity']
scope2_s =  clean_df['Scope 2 total KGCO2e']


predictions_scope2_df = pd.DataFrame({'Client Name':cliemtname,'Purchased Electricity':purchElec_s,
                               'Scope 2 total KGCO2e':scope2_s,
                               'Sklearn scope 2 Predictions':sklearn_scope2_predictions})



predictions_scope2_df['change'] = (predictions_scope2_df['Sklearn scope 2 Predictions'] - predictions_scope2_df['Scope 2 total KGCO2e'])/ predictions_scope2_df['Scope 2 total KGCO2e'] * 100


y_true = predictions_scope2_df['Scope 2 total KGCO2e'].values.tolist()
y_pred = predictions_scope2_df['Sklearn scope 2 Predictions'].values.tolist()

st.write("The mean absolute error for data used is: " ,mean_absolute_error(y_true, y_pred))
st.write("The mean absolute error for data used percentage is: " ,mean_absolute_percentage_error(y_true, y_pred))


with st.expander("See Model Data"):
    st.dataframe(predictions_scope2_df)

st.markdown("### Explanation of Scope 2 results")
st.write(
    """ A summary of the linear regression model for estimating scope 2 emmissions using Purchased Electricity.
The accuracy of predicting scope 2 emmisions based on Purchased Electricity is low , however, it is the most accurate model when compared to models for scope 1 and 3 emmisions. The main concern for this model is that values for purchased steam, heat and cooling are mostly missing/ 0 values. This missing data could potentially mean the model accuracy is misrepreseneted when applied to real world data.


Mean absolute Error: 7371.5774233583315

In the context of machine learning, absolute error refers to the magnitude of difference between the prediction of an observation and the true value of that observation. MAE takes the average of absolute errors for a group of predictions and observations as a measurement of the magnitude of errors for the entire group. MAE can also be referred as L1 loss function. As can be seen the mean absolute error for this model is somewhat high.


Mean Absolute percentage Error:3.6286663451654497e+18

Mean absolute percentage error measures the average magnitude of error produced by a model, or how far off predictions are on average. A MAPE value of 3.6286663451654497e+18 means that the average absolute percentage difference between the predictions and the actuals is 3.6286663451654497e+18. In other words, this model is somewhat accuarate at predicting scope 2 values based on the data provided, however, this could be linked to the values for purchased steam, heat and cooling being mostly missing/ 0 values. More complete data would need to be provided in order to determine the true accuracy of the model. """
)