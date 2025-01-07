import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Set up page configuration
st.set_page_config(
    page_title="Airbnb Listings Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("listings.csv")

df = load_data()

#df = preprocess_data(df)

# Sidebar options
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose a section:",
    ["Introduction", "EDA", "Model Training", "Conclusion"]
)

# Introduction Section
if option == "Introduction":
    st.title("Welcome to the Airbnb Listings Analysis App")
    st.write("This app provides an in-depth analysis of Airbnb listings and predicts their prices based on various features.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg", width=200)

# EDA Section
elif option == "EDA":
    st.title("Exploratory Data Analysis")
    eda_option = st.selectbox(
        "Choose an analysis:",
        [
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



        ]
    )

    if eda_option == "Summary Statistics":
        st.subheader("Summary Statistics")
        st.write(df.describe(include="all"))

    elif eda_option == "Missing Value Analysis":
        st.subheader("Missing Value Analysis")
        st.write(df.isnull().sum())

    
    elif eda_option == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

    elif eda_option == "Feature Distributions":
        st.markdown("### Feature Distributions (Histograms)")
        features = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month']
        for feature in features:
            st.markdown(f"**{feature.capitalize()} Distribution**")
            plt.figure()
            sns.histplot(df[feature], kde=True, bins=30, color="#3498db")
            st.pyplot(plt)

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
        sns.boxplot(x='neighbourhood', y='price', data=df)
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

    #elif eda_option == "Availability vs Price":
        #st.markdown("### Availability vs Price")
        #plt.figure()
        #sns.scatterplot(data=df, x='availability_365', y='price', alpha=0.5, color="#3498db")
        #plt.title("Availability vs Price")
        #st.pyplot(plt)

    elif eda_option == "Review Scores Distribution":
        st.markdown("### Review Scores Distribution")
        review_columns = [col for col in df.columns if "number_of_reviews" in col]
        for column in review_columns:
            if column in df.columns:
                plt.figure()
                sns.histplot(df[column], kde=True, bins=30, color="#9b59b6")
                plt.title(f"Distribution of {column.replace('_', ' ').capitalize()}")
                st.pyplot(plt)



# Model Training Section
elif option == "Model Training":
    #st.title("Model Training and Evaluation")

    # Preprocess the dataset

# Prediction Section
 #Data preprocessing
    st.header("Data Preprocessing")
    data = df.dropna(subset=['price'])  # Remove rows with missing target values
    # Select relevant features
    features = ["latitude", "longitude", "room_type", "neighbourhood_group"]
    df = df[features + ['price']]
    df = pd.get_dummies(df, columns=["room_type", "neighbourhood_group"], drop_first=True)

    # Split data into training and testing sets
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

    # Prediction
    st.header("Make Predictions")
    latitude = st.number_input("Latitude", value=1.35)
    longitude = st.number_input("Longitude", value=103.85)
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
    neighbourhood_group = st.selectbox("Neighbourhood Group", ["Central Region", "North Region", "East Region", "West Region", "North-East Region"])

    # Prepare input for prediction
    input_data = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "room_type_Private room": [1 if room_type == "Private room" else 0],
        "room_type_Shared room": [1 if room_type == "Shared room" else 0],
        "neighbourhood_group_North Region": [1 if neighbourhood_group == "North Region" else 0],
        "neighbourhood_group_East Region": [1 if neighbourhood_group == "East Region" else 0],
        "neighbourhood_group_West Region": [1 if neighbourhood_group == "West Region" else 0],
        "neighbourhood_group_North-East Region": [1 if neighbourhood_group == "North-East Region" else 0]
    })

    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:.2f}")


# Conclusion Section
elif option == "Conclusion":
    st.title("Conclusion")
    st.write("This app demonstrates how to analyze Airbnb listings data and predict prices based on selected features.")
    st.write("This project demonstrates a complete workflow from data exploration to model training and prediction. The flexibility to select features for predictions enhances the user experience, providing deeper insights into the model's performance """)
    
