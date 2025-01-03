# Data Science Project with Streamlit

This repository contains a data science project showcasing exploratory data analysis (EDA), preprocessing, machine learning model training, and visualization using a Streamlit web application. The dataset used for this project is a collection of Airbnb listings.

*Features of the Project*

1. **Exploratory Data Analysis (EDA)**
   - Summary statistics (mean, median, mode, etc.).
   - Missing value analysis.
   - Correlation heatmap (numeric features only).
   - Feature distribution visualizations.
   - Outlier detection using box plots.
   - Grouped analysis for additional insights.

2. **Data Preprocessing**
   - Handling missing values by removal.
   - Encoding categorical variables using one-hot encoding.
   - Normalizing and scaling features as required.
   - Splitting the dataset into training and testing subsets.

3. **Machine Learning Model**
   - A Random Forest Classifier is used as the machine learning model.
   - Evaluation of the model using metrics such as accuracy, classification report, and confusion matrix.
   - Analysis of feature importance for interpretability.

4. **Interactive Streamlit Application**
   - Displays EDA insights with interactive visualizations.
   - Includes a user-friendly interface for model results and evaluations.
   - Showcases key takeaways and feature importance visualizations.

## Folder Structure

```
├── app1.py               # Main Python script for the Streamlit application
├── listings.csv          # Dataset used for the analysis (replace with your file)
├── README.md             # Project documentation
└── requirements.txt      # Required Python libraries
```

## Requirements

To run this project, you need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit



## Insights from the Project

- **EDA:** Visualizations and statistical analysis reveal insights about the dataset's structure and relationships.
- **Preprocessing:** Efficient handling of missing values and encoding ensures readiness for model training.
- **Model Performance:** Random Forest Classifier achieved high accuracy, and feature importance analysis highlights key predictors.


