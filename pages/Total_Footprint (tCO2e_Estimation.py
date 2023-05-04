import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Total FootPrint", page_icon="📈")

st.markdown("# Estimating Total Footprint Emmissions")
st.sidebar.header("Total Foot print Estimation")

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

scope3 = clean_df['Scope 3 total KGCO2e']
totalcarb = clean_df['Total Footprint (tCO2e)']

scope3_np = scope3.to_numpy()
totalcarb_np = totalcarb.to_numpy()

sklearn_model = LinearRegression().fit(scope3_np.reshape((109,1)), totalcarb_np)
sklearn_total_predictions = sklearn_model.predict(scope3_np.reshape((109, 1)))

number = st.number_input('Insert Scope 3 Total')
st.write('The current number Scope 3 Total Emmissions is ', number)

first_row = np.array([number])

scope3_user_pred = sklearn_model.predict(first_row.reshape(1, -1))
st.write('The estimated Total Footprint (tCO2e) of emmissions is ', scope3_user_pred.flat[0] )


cliemtname = clean_df['Client Name']
scope3_s = clean_df['Scope 3 total KGCO2e']
total_s =  clean_df['Total Footprint (tCO2e)']


total_predictions_df = pd.DataFrame({'Client Name':cliemtname,'Scope 3 total KGCO2e':scope3_s,
                               'Total Footprint (tCO2e)':total_s,
                               'Sklearn total Predictions':sklearn_total_predictions})

total_predictions_df['change'] = (total_predictions_df['Sklearn total Predictions'] - total_predictions_df['Total Footprint (tCO2e)'])/ total_predictions_df['Total Footprint (tCO2e)'] * 100


y_true = total_predictions_df['Total Footprint (tCO2e)'].values.tolist()
y_pred = total_predictions_df['Sklearn total Predictions'].values.tolist()

st.write("The mean absolute error for data used is: " ,mean_absolute_error(y_true, y_pred))
st.write("The mean absolute error for data used percentage is: " ,mean_absolute_percentage_error(y_true, y_pred))


with st.expander("See Model Data"):
    st.dataframe(total_predictions_df)

st.markdown("### Explanation of Total Emmissions")
st.write(
    """A summary of the linear regression model for total footprint emissions using total scope 3 emissions.
The accuracy of predicting total scope value with scope 3 total emissions values is somewhat reasonable, however, this model should only be used as a proof of concept demonstrator, as more data is required to determine the true accuracy of the mode.

Mean absolute Error: 264.48651441981775

In the context of machine learning, absolute error refers to the magnitude of difference between the prediction of an observation and the true value of that observation. MAE takes the average of absolute errors for a group of predictions and observations as a measurement of the magnitude of errors for the entire group. MAE can also be referred as L1 loss function. As can be seen the mean absolute error for this model is moderate.

Mean Absolute percentage Error:2.1967084385094404

Mean absolute percentage error measures the average magnitude of error produced by a model, or how far off predictions are on average. A MAPE value of 2.1967084385094404 means that the average absolute percentage difference between the predictions and the actuals is 2.1967084385094404
. In other words, this model is somewhat accurate at predicting total emmision values based on the data provided, however, this could be linked to missing data values. More complete data would need to be provided in order to determine the true accuracy of the model.

The process of creating the linear regression model for scope total FootPrint estimations will now be explained."""
)