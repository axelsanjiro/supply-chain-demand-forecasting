import streamlit as st
import pandas as pd
import plotly.express as px

# Configure Page
st.set_page_config(
    page_title="Exploratory Data Analysis", 
    page_icon="📊", 
    layout="wide"
    )

st.title("📊 Exploratory Data Analysis (EDA)")
st.info("""
**Business Context:**  
When managing a supply chain, a business has to make a crucial decision regarding how much inventory to stock based on historical data and market factors. Two types of risks are associated with this inventory decision:

* If the actual demand is high, then **understocking** the product results in a loss of potential sales and decreased customer trust.
* If the actual demand is low, then **overstocking** the product results in a financial loss to the business due to high warehousing costs and potential spoilage.
""")

# Button to Description Page
st.write("For further information regarding the dataset features visit:")
if st.button("Dataset Description"):
    st.switch_page("pages/2_Description_Page.py")

st.markdown("---")

# Data Loading
# @st.cache_data prevents Streamlit from reloading the large CSV file on every interaction
@st.cache_data
def load_data():
    try:
        # Read the raw data
        df = pd.read_csv("dataset/train.csv")
        
        # Convert 'week' to datetime format
        df['date'] = pd.to_datetime(df['week'], format='%y/%m/%d')
        
        # Basic Feature Engineering for EDA purposes
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_name'] = df['date'].dt.day_name()
        
        return df
    except Exception as e:
        return None

with st.spinner('Loading dataset and generating visualizations...'):
    df = load_data()

if df is None:
    st.error("Dataset not found. Please ensure 'train.csv' is placed inside the 'dataset' folder.")
    st.stop()

# Data Overview
st.subheader("1. Data Overview")
st.write("Here are the first 5 rows of our historical sales data:")
st.dataframe(df.head(), use_container_width=True)
st.success(f"The loaded dataset contains **{df.shape[0]:,} rows** and **{df.shape[1]} columns**.")

st.markdown("---")

# Visualization: Overall Sales Trend
st.subheader("2. Overall Sales Trend")
st.write("This chart illustrates the total weekly sales volume across all stores and items over time. Notice the recurring yearly patterns.")

# Aggregate weekly sales
weekly_sales = df.groupby('week')['units_sold'].sum().reset_index()

fig_trend = px.line(
    weekly_sales, x='week', y='units_sold', 
    title='Total Weekly Sales (Units Sold over time)',
    labels={'week': 'Week', 'units_sold': 'Total Units Sold'}
)
fig_trend.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig_trend, use_container_width=True)

# Visualization: Promotions and Stores
col1, col2 = st.columns(2)

with col1:
    st.subheader("3. Impact of Promotions")
    st.write("How does featuring a product affect its sales?")
    
    # Agregate sales by promotion status
    promo_sales = df.groupby('is_featured_sku')['units_sold'].mean().reset_index()
    promo_sales['is_featured_sku'] = promo_sales['is_featured_sku'].map({0: 'Not Featured', 1: 'Featured'})
    
    fig_promo = px.bar(
        promo_sales, x='is_featured_sku', y='units_sold', 
        title='Average Units Sold: Featured vs Non-Featured',
        labels={'is_featured_sku': 'Promotion Status', 'units_sold': 'Avg Units Sold'},
        color='is_featured_sku'
    )
    st.plotly_chart(fig_promo, use_container_width=True)

with col2:
    st.subheader("4. Top Store Performance")
    st.write("Which stores contribute the most to total sales?")
    
    # top 10 stores by total sales
    store_sales = df.groupby('store_id')['units_sold'].sum().reset_index()
    top_stores = store_sales.sort_values(by='units_sold', ascending=False).head(10)
    top_stores['store_id'] = top_stores['store_id'].astype(str)
    
    fig_store = px.bar(
        top_stores, x='store_id', y='units_sold',
        title='Top 10 Stores by Total Sales',
        labels={'store_id': 'Store ID', 'units_sold': 'Total Units Sold'},
        color='units_sold', color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_store, use_container_width=True)