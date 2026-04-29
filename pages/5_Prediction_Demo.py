import streamlit as st
import pandas as pd
import numpy as np

# Configure Page
st.set_page_config(
    page_title="Prediction Demo",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Real-Time Demand Prediction")
st.markdown("---")

# Safety Check: Ensure the model exists
if 'rf_model' not in st.session_state or st.session_state['rf_model'] is None:
    st.warning("The model hasn't been trained yet. Please go to the **Train Model** page first.")
    if st.button("⬅️ Go to Train Model"):
        st.switch_page("pages/4_Train_Your_Model.py")
    st.stop()

rf_model = st.session_state['rf_model']
expected_features = st.session_state['X_train'].columns.tolist()

st.write("""
Simulate a business scenario by adjusting the pricing and promotional strategies below. 
The AI will forecast the expected demand (Units Sold) so you can optimize your inventory and avoid both spoilage and stockouts.
""")

# Layout for Inputs and Outputs
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Scenario Setup")
    st.write("Adjust the business levers:")
    
    # Basic Identifiers
    c_store, c_sku = st.columns(2)
    with c_store:
        store_id = st.number_input("Store ID", min_value=8000, max_value=9999, value=8091, step=1)
    with c_sku:
        sku_id = st.number_input("Product SKU ID", min_value=200000, max_value=300000, value=216418, step=1)
    
    st.markdown("##### Pricing Strategy")
    base_price = st.number_input("Base Price ($)", min_value=1.0, value=150.0, step=1.0)
    total_price = st.number_input("Discounted/Checkout Price ($)", min_value=1.0, value=135.0, step=1.0)
    
    st.markdown("##### Promotional Strategy")
    c_promo1, c_promo2 = st.columns(2)
    with c_promo1:
        is_featured = st.checkbox("Featured Product (Flyer/Ad)", value=True)
    with c_promo2:
        is_display = st.checkbox("Prominent Store Display", value=True)
        
    st.markdown("##### Temporal Factors")
    c_time1, c_time2 = st.columns(2)
    with c_time1:
        month = st.slider("Month", 1, 12, 7)
    with c_time2:
        year = st.selectbox("Year", [2013, 2014, 2015, 2016])

with col2:
    st.subheader("AI Demand Forecast")

    # Dynamic Dictionary Building
    # This ensures we only feed the model the exact columns it was trained on
    input_data = {}
    if 'store_id' in expected_features: input_data['store_id'] = store_id
    if 'sku_id' in expected_features: input_data['sku_id'] = sku_id
    if 'total_price' in expected_features: input_data['total_price'] = total_price
    if 'base_price' in expected_features: input_data['base_price'] = base_price
    if 'is_featured_sku' in expected_features: input_data['is_featured_sku'] = 1 if is_featured else 0
    if 'is_display_sku' in expected_features: input_data['is_display_sku'] = 1 if is_display else 0
    if 'month' in expected_features: input_data['month'] = month
    if 'year' in expected_features: input_data['year'] = year
    if 'discount_amount' in expected_features: input_data['discount_amount'] = base_price - total_price

    # Fallback safety: fill missing features with 0 to prevent crashes
    for col in expected_features:
        if col not in input_data:
            input_data[col] = 0
            
    # Convert to DataFrame and enforce column order exactly like the training data
    input_df = pd.DataFrame([input_data])[expected_features]
    
    st.write("**Data sent to the model:**")
    st.dataframe(input_df, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction Execution
    if st.button("Forecast Demand Now", type="primary", use_container_width=True):
        with st.spinner("AI is calculating the optimum inventory..."):
            
            # Predict
            prediction = rf_model.predict(input_df)[0]
            
            # Post-processing: Demand can't be negative, and usually we round up for inventory
            final_prediction = max(0, int(np.ceil(prediction)))
            
            # Custom UI for the final result
            st.markdown(f"""
            <div style="background-color: #262730; padding: 30px; border-radius: 15px; border: 2px solid #FF4B4B; text-align: center; margin-top: 10px;">
                <h3 style="color: #FAFAFA; margin-bottom: 0;">Predicted Demand</h3>
                <h1 style="color: #FF4B4B; font-size: 75px; margin-top: 0px; margin-bottom: 0px;">{final_prediction}</h1>
                <p style="font-size: 18px; color: #A0A0A0; margin-top: 0px;">units required</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            # Business Actionable Insight
            st.success(f"**Actionable Insight:** To prevent stockouts for SKU {sku_id} at Store {store_id} based on your ${base_price - total_price} discount strategy, dispatch at least **{final_prediction} units** to the warehouse this week.")

# 6. Conclusion
st.markdown("---")
st.caption("End of the Supply Chain Demand Forecasting pipeline. Thank you for exploring!")