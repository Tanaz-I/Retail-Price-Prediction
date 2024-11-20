import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import shap
import os



# Load datasets
kohl = pd.read_csv("Kohl's.csv")
amazon = pd.read_csv("Amazon.csv")
foot = pd.read_csv("Foot Locker.csv")
sports = pd.read_csv("Sports Direct.csv")
walmart = pd.read_csv("Walmart.csv")
west = pd.read_csv("West Gear.csv")

retailer_dfs = {'Kohl\'s': kohl, 'Amazon': amazon, 'Foot Locker': foot, 'Sports Direct': sports, 'Walmart': walmart, 'West Gear': west}

# Streamlit UI setup
st.title("Retailer Sales Price Optimization")
st.write("This app optimizes the price per unit for different retailers.")

for retailer, df in retailer_dfs.items():
    st.subheader(f"{retailer} Price optimization")

    # Drop irrelevant columns and prepare features and target
    x = df.drop(columns=['Retailer', 'Invoice Date', 'Price per Unit', 'Units Sold', 'Total Sales', 'lag_price', 'Month', 'Day', 'Week', 'profit_percentage'])
    y = df['Price per Unit']

    # Split data into train, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(x_train, y_train)

    # Predict and evaluate on validation data
    y_val_pred = rf_regressor.predict(x_val)
    mse_val = mean_squared_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)

    st.write(f"Validation MSE for {retailer}: {mse_val:.2f}")
    st.write(f"Validation R2 Score for {retailer}: {r2_val:.2f}")

    explainer = shap.TreeExplainer(rf_regressor)
    shap_values=explainer(x_val)

    mean_shap_values = np.abs(shap_values.values).mean(axis=0)  # Access .values correctly

    # Find the overall highest contributing feature
    overall_highest_index = np.argmax(mean_shap_values)
    overall_highest_feature = x.columns[overall_highest_index]
    overall_highest_value = mean_shap_values[overall_highest_index]

    st.write(f"Overall highest contributing feature for {retailer}: {overall_highest_feature} with average absolute SHAP value: {overall_highest_value:.2f}")

    st.write("SHAP Summary Plot")
    fig_summary = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, x_val)
    st.pyplot(fig_summary)
    st.write("") 

    # Display predictions for 3 random points in the validation set
    random_sample = x_val.sample(3, random_state=42)
    st.write("Predictions for 3 random points in the validation set:")
    for i, row in random_sample.iterrows():
        
        actual_price = y_val.loc[i]
        predicted_price = rf_regressor.predict([row])[0]
        st.write(f"Actual Price: {actual_price:.2f}, Predicted Price: {predicted_price:.2f}")
        index_in_val = x_val.index.get_loc(i)    
    
        shap_values_single = shap.Explanation(
        values=shap_values.values[index_in_val], 
        base_values=shap_values.base_values[index_in_val], 
        data=row
        )
        abs_shap_values = np.abs(shap_values_single.values)  
        highest_contrib_index = np.argmax(abs_shap_values)  
        highest_contrib_feature = x_val.columns[highest_contrib_index] 
        highest_contrib_value = shap_values_single.values[highest_contrib_index]  
        st.write(f"Highest contributing feature: {highest_contrib_feature} with SHAP value: {highest_contrib_value:.2f}")
        st.write(f"SHAP Explanation for Prediction {i+1}")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values_single, max_display=10)
        st.pyplot(fig)
        


        
       
