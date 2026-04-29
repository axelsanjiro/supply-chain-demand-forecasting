import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure Page
st.set_page_config(
    page_title="Train Your Model", 
    page_icon="🧠", 
    layout="wide"
)

st.title("🧠 Train Your Machine Learning Model")
st.markdown("---")

# Check if Preprocessing is Done
if 'is_preprocessed' not in st.session_state or not st.session_state['is_preprocessed']:
    st.warning("You haven't preprocessed the data yet. Please go to the **Preprocessing** page to prepare the data first.")
    if st.button("⬅️ Go to Preprocessing"):
        st.switch_page("pages/3_Data_Preprocessing.py")
    st.stop() 

st.write("""
Now that our time-series data is clean and split chronologically, it's time to train the **Random Forest Regressor**. 
You can tune the model's hyperparameters below to see how it affects the forecasting accuracy.
""")

# Retrieve Data from Memory
X_train = st.session_state['X_train']
X_test = st.session_state['X_test']
y_train = st.session_state['y_train']
y_test = st.session_state['y_test']

# Hyperparameter Tuning
st.subheader("1. Hyperparameter Tuning")
col1, col2 = st.columns(2)

with col1:
    n_estimators = st.slider(
        "Number of Trees (n_estimators)", 
        min_value=10, max_value=200, value=50, step=5,
        help="The number of decision trees in the forest. More trees generally increase accuracy but take longer to compute."
    )
    st.info("**Optimal Guide:** The sweet spot is usually between **100 - 150**. Going higher often hits a plateau where accuracy stops improving, but the app runs slower.")
with col2:
    max_depth = st.slider(
        "Maximum Depth (max_depth)", 
        min_value=5, max_value=50, value=15, step=5,
        help="The maximum depth of each tree. Limits the complexity to prevent the model from memorizing the training data (overfitting)."
    )
    st.info("**Optimal Guide:** The ideal depth is typically between **10 - 20**. Too low causes *underfitting* (fails to learn patterns), while too high causes *overfitting* (memorizes training data but fails on future data).")

st.markdown("---")

# Training Engine
if st.button("Train Random Forest Model", use_container_width=True):
    with st.spinner("Training the model... This might take a few seconds."):
        
        # Initialize and Train the Model
        # n_jobs=-1 makes it train faster by using all CPU cores
        rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)

        # Predict on Future (Test) Data
        y_pred = rf_model.predict(X_test)
        
        # Save model to memory for the final demo page
        st.session_state['rf_model'] = rf_model
        st.session_state['is_trained'] = True
        
        # Calculate Error Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.session_state['model_mae'] = mae
        st.session_state['model_rmse'] = rmse
        st.session_state['model_r2'] = r2
        
        st.success("Model trained successfully!")

# Display Evaluation Metrics
        st.subheader("2. Model Evaluation Metrics")
        st.write("How well did the model forecast the unseen 'future' data?")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Score", f"{r2:.4f}", help="Accuracy percentage. Closer to 1.0 is better.")
        m_col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}", help="On average, the model's prediction is off by this many units.")
        m_col3.metric("Root Mean Sq. Error (RMSE)", f"{rmse:.2f}", help="Similar to MAE but gives higher penalty to large mistakes.")

# Visualizing Results
        st.subheader("3. Visualizing Performance")
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            # Actual vs Predicted Plot (Taking a subset of 100 rows so the line chart isn't too messy)
            st.write("**Actual vs Predicted Demand (Subset of Test Data)**")
            chart_data = pd.DataFrame({'Actual Demand': y_test.values, 'Predicted Demand': y_pred})
            fig_pred = px.line(
                chart_data.head(100), 
                labels={'index': 'Time Sequence (Future Weeks)', 'value': 'Units Sold'},
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
        with v_col2:
            # Feature Importance Plot
            st.write("**Feature Importance**")
            importances = rf_model.feature_importances_
            fi_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=True)
            
            fig_fi = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Viridis'
            )
            fig_fi.update_layout(showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)

# Navigation to Final Page (outside the training button conditional)
if st.session_state.get('is_trained', False):
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav2:
        if st.button("Proceed to Prediction Demo ➡️", use_container_width=True):
            st.switch_page("pages/5_Prediction_Demo.py")