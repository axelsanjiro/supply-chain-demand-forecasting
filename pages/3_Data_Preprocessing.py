import streamlit as st
import pandas as pd

# Configure Page
st.set_page_config(
    page_title="Data Preprocessing",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Data Preprocessing & Feature Engineering")
st.markdown("---")

st.write("""
Before feeding data into our Random Forest model, we need to clean it and create meaningful features. 
In this step, you can configure how the data is processed and split for training.
""")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset/train.csv")
        df['week'] = pd.to_datetime(df['week'], format='%d/%m/%y')
        # Crucial for Time Series: Sort values chronologically
        df = df.sort_values(by='week').reset_index(drop=True)
        return df
    except Exception:
        return None

df = load_data()

if df is None:
    st.error("⚠️ Dataset not found. Please ensure 'train.csv' is placed inside the 'data' folder.")
    st.stop()

# Data Overview
st.subheader("Data Overview")
st.write("Here are the first 5 rows of our historical sales data:")
st.dataframe(df.head(), use_container_width=True)

# Interactive Preprocessing Configuration
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Feature Engineering")
    st.write("Extract new information from existing columns to help the model learn better patterns.")
    
    # User toggles for feature engineering
    extract_date = st.checkbox("Extract 'Month' and 'Year' from 'week' column", value=True)
    calculate_discount = st.checkbox("Calculate 'discount_amount' (base_price - total_price)", value=True)
    drop_id = st.checkbox("Drop 'record_ID' (Identifiers do not hold predictive value)", value=True)
    
with col2:
    st.subheader("2. Train-Test Split (Chronological)")
    st.write("""
    Since this is time-series data, we cannot split it randomly. 
    We must train the model on **past data** and test it on **future data** to simulate real-world forecasting.
    """)
    
    # Slider diperlebar agar user bisa bereksperimen, tapi tetap aman dari crash
    test_size = st.slider("Select Test Data Percentage (%)", min_value=5, max_value=50, value=20, step=1)
    
    # Pesan edukasi dinamis berdasarkan input user
    if test_size < 10 or test_size > 30:
        st.warning("**Note:** While you can use this split, the industry standard for test size is typically between 10% - 30% to ensure the model has enough data to learn, while still leaving enough to be tested properly.")
    else:
        st.success("**Optimal Split!** 10% - 30% is the ideal range to balance model learning and evaluation.")

st.markdown("---")

# Execution Engine
if st.button("Apply Preprocessing & Split Data", use_container_width=True):
    with st.spinner("Processing data..."):
        processed_df = df.copy()
        
        # Apply selected Feature Engineering
        if extract_date:
            processed_df['month'] = processed_df['week'].dt.month
            processed_df['year'] = processed_df['week'].dt.year
            # Drop the original datetime column as Random Forest only takes numbers
            processed_df = processed_df.drop('week', axis=1)
            
        if calculate_discount:
            processed_df['discount_amount'] = processed_df['base_price'] - processed_df['total_price']
            
        if drop_id and 'record_ID' in processed_df.columns:
            processed_df = processed_df.drop('record_ID', axis=1)
            
        # Define Features (X) and Target (y)
        X = processed_df.drop('units_sold', axis=1)
        y = processed_df['units_sold']
        
        # Calculate split index based on the slider value
        split_idx = int(len(processed_df) * (1 - (test_size / 100)))
        
        # Split the data chronologically
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Save datasets to Streamlit Session State for the next page
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['is_preprocessed'] = True
        
        st.success("Preprocessing successful! Data has been split and saved to memory.")
        
        # Display Results
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.info(f"**Training Data (Past):** {X_train.shape[0]:,} rows")
            st.dataframe(X_train.head(), use_container_width=True)
        with res_col2:
            st.info(f"**Testing Data (Future):** {X_test.shape[0]:,} rows")
            st.dataframe(X_test.head(), use_container_width=True)

# Navigation to the next page (outside the button conditional)
if st.session_state.get('is_preprocessed', False):
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    
    with col_nav2:
        if st.button("Proceed to Train Model ➡️", use_container_width=True):
            st.switch_page("pages/4_Train_Your_Model.py")