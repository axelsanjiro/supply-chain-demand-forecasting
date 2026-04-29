import streamlit as st

# Configurasi Page
st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

# Initiate Session State
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None

if 'rf_model' not in st.session_state:
    st.session_state['rf_model'] = None

# Main Page Content
st.title("Supply Chain Demand Forecasting System")

col1, col2 = st.columns([1, 1])

with col1:
    st.image("assets/supply_chain.jpg", use_container_width=True)

with col2:
    st.write("""
    Welcome to the Supply Chain Demand Forecasting Application! 

    This interactive application is built using the **Random Forest** algorithm to predict commodity demand based on historical sales, pricing, and promotional factors. 
    This project aims to demonstrate a data-driven solution to a classic problem in the logistics and retail industries: **preventing overstock and avoiding understock.**
    """)
st.markdown("---")

st.info("Please use the menu in the left sidebar to navigate the application.")

# Navigation Links
st.markdown("### Start Exploring:")
col_nav1, col_nav2 = st.columns(2)

with col_nav1:
    st.write("Understand the dataset characteristics:")
    st.page_link("pages/1_Exploratory_Data_Analysis.py", label="View Exploratory Data Analysis (EDA)", icon="📊")
    
with col_nav2:
    st.write("Test the model interactively:")
    st.page_link("pages/5_Prediction_Demo.py", label="Try the Prediction Demo", icon="🚀")

st.markdown("---")
st.caption("Developed as a Machine Learning portfolio project.")