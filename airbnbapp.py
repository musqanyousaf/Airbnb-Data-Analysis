import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np


# Set up page configuration
st.set_page_config(
    page_title="Airbnb Listings Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the dataset
file_path = "listings.csv"
df = pd.read_csv(file_path)



# Preprocessing
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce').fillna(pd.to_datetime("2020-01-01"))
df.dropna(subset=['name', 'price'], inplace=True)

# Feature engineering
df['days_since_last_review'] = (pd.to_datetime('today') - df['last_review']).dt.days
df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=True)

# Outlier removal
df = df[(df['price'] >= 10) & (df['price'] <= df['price'].quantile(0.99))]

# Feature selection and target variable
features = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count']
target_col = "availability_365"
df[target_col] = df[target_col].apply(lambda x: 1 if x == 365 else 0)  # Convert to binary target

X = df[features]
y = df[target_col]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline for scaling and model training
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# Train the model
@st.cache_data
def train_model(X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

pipeline = train_model(X_train, y_train)

# Sidebar options
st.sidebar.header("Select Task")
option = st.sidebar.radio(
    "Choose the task to perform:",
    (
        "Introduction",
        "Perform EDA",
        "Train and Predict Availability",
        "Conclusion",
    ),
)

# Introduction Section
if option == "Introduction":
    st.title("Welcome to the Airbnb Listings Analysis App")
    st.write("This app provides an in-depth analysis of Airbnb listings and predicts their prices based on various features.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg", width=200)

# EDA Section

# Task: EDA
elif option == "Perform EDA":
    st.markdown("## Exploratory Data Analysis (EDA)")
    eda_option = st.radio(
        "Choose the analysis to display:",
        (
            "Summary Statistics",
            "Missing Value Analysis",
            "Correlation Heatmap",
            "Feature Distributions",
            "Room Type Count",
            "Neighborhood Group Count",
            "Pairwise Relationships",
            "Box Plot for Outliers",
            "Trend Over Time",
            "Grouped Aggregations",
            "Availability Analysis",
            "Price vs Neighborhood Relationship",
            "Review Scores Distribution",
            "Price vs Longitude",
            "Availability vs Price",
            "Number of Reviews vs Price"

        ),
    )

    if eda_option == "Summary Statistics":
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe(include="all"))

    elif eda_option == "Missing Value Analysis":
        st.markdown("### Missing Value Analysis")
        st.dataframe(df.isnull().sum())

    elif eda_option == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

    elif eda_option == "Feature Distributions":
        st.markdown("### Feature Distributions")
        features = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month']
        for feature in features:
            plt.figure()
            sns.histplot(df[feature], kde=True, bins=30)
            st.pyplot(plt.gcf())
            plt.clf()

    elif eda_option == "Room Type Count":
        st.markdown("### Room Type Count")
        plt.figure()
        sns.countplot(data=df, x="room_type", palette="Set2")
        st.pyplot(plt)

    elif eda_option == "Neighborhood Group Count":
        st.markdown("### Neighborhood Group Count")
        plt.figure()
        sns.countplot(data=df, x="neighbourhood_group", palette="Set3")
        st.pyplot(plt)

    elif eda_option == "Pairwise Relationships":
        st.markdown("### Pairwise Relationships")
        pairplot_fig = sns.pairplot(df[['price', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews']].dropna())
        st.pyplot(pairplot_fig)

    elif eda_option == "Box Plot for Outliers":
        st.markdown("### Box Plot for Room Type vs Price")
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="room_type", y="price")
        st.pyplot(plt)

    elif eda_option == "Trend Over Time":
        st.markdown("### Trend of Reviews Over Time")
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
        df_time = df.dropna(subset=['last_review']).groupby(df['last_review'].dt.to_period("M")).agg({'number_of_reviews': 'sum'})
        plt.figure()
        plt.plot(df_time.index.to_timestamp(), df_time['number_of_reviews'], marker='o', color="#2ecc71")
        st.pyplot(plt)

    elif eda_option == "Grouped Aggregations":
        st.markdown("### Mean Price by Neighbourhood Group")
        grouped_price = df.groupby("neighbourhood_group")['price'].mean()
        st.bar_chart(grouped_price)

    elif eda_option == "Availability Analysis":
        st.markdown("### Availability Over 365 Days")
        plt.figure()
        sns.histplot(df["availability_365"], kde=False, bins=20, color='#9b59b6')
        st.pyplot(plt)

    elif eda_option == "Price vs Neighborhood Relationship":
        st.markdown("### Price vs Neighborhood Relationship")
        plt.figure(figsize=(12, 6))
        sns.bar(x='neighbourhood', y='price', data=df)
        plt.xticks(rotation=90)
        plt.title("Price Distribution Across Neighborhoods")
        st.pyplot(plt)
    
    elif eda_option == "Price vs Longitude":
        st.markdown("### Price vs Longitude")
        plt.figure()
        sns.scatterplot(data=df, x='longitude', y='price', alpha=0.5, color="#e74c3c")
        plt.title("Price vs Longitude")
        st.pyplot(plt)

    elif eda_option == "Number of Reviews vs Price":
        st.markdown("### Number of Reviews vs Price")
        plt.figure()
        sns.scatterplot(data=df, x='number_of_reviews', y='price', alpha=0.5, color="#8e44ad")
        plt.title("Number of Reviews vs Price")
        st.pyplot(plt)

    elif eda_option == "Availability vs Price":
        st.markdown("### Availability vs Price")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=pd.cut(df['availability_365'], bins=5, labels=["Very Low", "Low", "Medium", "High", "Very High"]), y='price', palette="Set2")
        plt.title("Availability vs Price")
        plt.xlabel("Availability Categories")
        plt.ylabel("Price")
        st.pyplot(plt)


# Task: Train and Predict Availability
elif option == "Train and Predict Availability":
    st.markdown("## Predict Availability for 365 Days")
    
    # Model accuracy
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # User input for prediction
    st.sidebar.header("Enter Airbnb Property Details")
    price = st.sidebar.number_input("Price", min_value=0, step=1)
    min_nights = st.sidebar.number_input("Minimum Nights", min_value=1, step=1)
    reviews = st.sidebar.number_input("Number of Reviews", min_value=0, step=1)
    reviews_per_month = st.sidebar.number_input("Reviews per Month", min_value=0.0, step=0.1)
    host_listings = st.sidebar.number_input("Number of Listings by Host", min_value=0, step=1)

    # Prepare input for prediction
    user_input = pd.DataFrame({
        'price': [price],
        'minimum_nights': [min_nights],
        'number_of_reviews': [reviews],
        'reviews_per_month': [reviews_per_month],
        'calculated_host_listings_count': [host_listings]
    })

    if st.button("Predict"):
        # Standardize input and predict
        prediction = pipeline.predict(user_input)
        prediction_proba = pipeline.predict_proba(user_input)[0]

        if prediction[0] == 1:
            st.error(f"This property is predicted not to be available for 365 days with a probability of {prediction_proba[1] * 100:.2f}%.")
        else:
            st.success(f"This property is predicted to be available for 365 days with a probability of {prediction_proba[0] * 100:.2f}%.")

# Task: Conclusion
elif option == "Conclusion":
    st.markdown("## Conclusion")
    st.markdown("""
        This project demonstrates a full workflow from data exploration to predictive modeling for Airbnb listings. 
        It provides insights into property availability based on various features.
    """)
    st.write("This project demonstrates a complete workflow from data exploration to model training and prediction. The flexibility to select features for predictions enhances the user experience, providing deeper insights into the model's performance """)

