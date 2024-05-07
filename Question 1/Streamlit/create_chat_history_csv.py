"""
Question 1 QnA with Streamlit - Creating Chat History CSV
"""

# Import all the necessary libraries
import pandas as pd
import os

##############
# WARNING! Only run this cell once to create the static folder and the history dataframe
# 
# Rerun this cell to clear the history dataframe
##############
# create static folder
if not os.path.exists('static'):
    os.makedirs('static')
# create df for history and save to csv
df = pd.DataFrame(columns=['entity', 'message'])
df.to_csv('static/df_history.csv', index=False)