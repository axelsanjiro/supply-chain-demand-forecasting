import streamlit as st
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Dataset Description", 
    page_icon="📝", 
    layout="wide"
    )

st.title("📝 Dataset Description & Dictionary")
st.markdown("---")

st.write("""
Before building our Machine Learning model, it is crucial to understand the variables (features) we are working with. 
This page serves as a comprehensive guide to the Supply Chain Demand Forecasting dataset.
""")

tab1, tab2 = st.tabs(["📖 Data Dictionary", "🔍 Raw Data Preview"])

with tab1:
    st.subheader("Feature Descriptions")
    st.write("Below is the detail of each column available in the raw dataset:")
    
    st.markdown("""
    | Feature Name | Data Type | Description |
    | :--- | :--- | :--- |
    | **record_ID** | `Identifier` | Unique ID for each row in the dataset. |
    | **week** | `Datetime` | The starting date of the week (Format: DD/MM/YY). |
    | **store_id** | `Categorical` | Unique identifier for the physical store/warehouse. |
    | **sku_id** | `Categorical` | Unique identifier for the product (Stock Keeping Unit). |
    | **total_price** | `Numerical` | The price at which the product was actually sold in that week. |
    | **base_price** | `Numerical` | The base/regular price of the product. |
    | **is_featured_sku** | `Binary` | `1` if the product was featured (e.g., in a promotional flyer), `0` otherwise. |
    | **is_display_sku** | `Binary` | `1` if the product was displayed prominently in the store, `0` otherwise. |
    | **units_sold** | `Numerical` | **[TARGET VARIABLE]** The total number of units sold for that specific SKU at that store during the week. |
    """)

    st.markdown("<br>", unsafe_allow_html=True) 
    
    with st.expander("💡 See Planned Feature Engineering", expanded=False):
        st.write("""
        During the **Preprocessing Phase**, we will engineer new features to help the Random Forest model capture pricing dynamics and seasonal trends:
        * **discount_amount**: Calculated as `base_price - total_price` to identify the direct impact of price cuts.
        * **month & year**: Extracted from the `week` column since algorithms cannot process raw date formats mathematically.
        """)

with tab2:
    st.subheader("Raw Data Preview")
    st.write("Here is a quick look at the first few rows of the dataset before any modifications:")
    
    # load only the first 100 rows for preview to optimize performance
    @st.cache_data
    def load_raw_data():
        try:
            return pd.read_csv("dataset/train.csv", nrows=100)
        except Exception:
            return None
            
    df_preview = load_raw_data()
    
    if df_preview is not None:
        # Menampilkan dataframe dengan UI Streamlit yang interaktif
        st.dataframe(df_preview, use_container_width=True)
    else:
        st.error("⚠️ Dataset not found. Please ensure 'train.csv' is inside the 'dataset' folder.")

st.markdown("---")

# Navigation Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("⬅️ Back to EDA"):
        st.switch_page("pages/1_Exploratory_Data_Analysis.py")
        
with col2:
    # Memposisikan tombol "Next" di sebelah kanan
    st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
    if st.button("Proceed to Preprocessing ➡️"):
        st.switch_page("pages/3_Data_Preprocessing.py")
    st.markdown('</div>', unsafe_allow_html=True)