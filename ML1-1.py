import pandas as pd
import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("real_estate_dataset.csv")

# Load the trained model
with open("linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# Sidebar navigation
def sidebar_navigation():
    st.sidebar.image("Logo.jpg", use_column_width=True)  # Replace with your image path
    section = st.sidebar.radio("Navigation", ["EDA", "Insights", "Model"])
    return section

# EDA Section
def eda_section(df):
    st.header("Exploratory Data Analysis")
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Dataset Description")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# Insights Section
# Insights Section
def insights_section(df):
    st.header("Insights")
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    # Distribution of Prices
    st.subheader("Distribution of Prices")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], kde=True, color="blue")
    st.pyplot(plt)
    
    # Add more insights here
    

    # Scatter Plot of Price vs Square_Feet
    st.subheader("Price vs Square Feet (colored by Number of Bedrooms)")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Square_Feet', y='Price', hue='Num_Bedrooms')
    plt.title('Square Feet vs Price (Hue: Number of Bedrooms)')
    st.pyplot(plt)

# Model Section
# Model Section
def model_section(model):
    st.header("Predict House Price")
    st.write("Use the sliders below to input your preferences, and the model will predict the price.")

    # User input sliders for number of bedrooms and square feet
    num_bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, step=1, value=3)
    square_feet = st.slider("Square Feet", min_value=500, max_value=10000, step=100, value=1500)

    # Dropdowns for other features (optional)
    st.subheader("Optional Features")
    
    # Use dropdowns for other features, setting them to None if no value is selected
    has_garden = st.selectbox("Has Garden?", ["Select", "Yes", "No"], index=0)
    has_pool = st.selectbox("Has Pool?", ["Select", "Yes", "No"], index=0)
    garage_size = st.selectbox("Garage Size?", ["Select", "1", "2", "3", "4", "5"], index=0)
    distance_to_center = st.selectbox("Distance to City Center (miles)?", ["Select", "0-5", "5-10", "10-20"], index=0)
    
    # Set default values if no selection is made
    has_garden = 1 if has_garden == "Yes" else 0
    has_pool = 1 if has_pool == "Yes" else 0
    garage_size = int(garage_size) if garage_size != "Select" else 0
    distance_to_center = {"0-5": 5, "5-10": 7, "10-20": 15}.get(distance_to_center, 0)

    # Prepare input for prediction with all expected features
    input_data = pd.DataFrame({
        "Num_Bedrooms": [num_bedrooms],
        "Square_Feet": [square_feet],
        "Has_Garden": [has_garden],
        "Has_Pool": [has_pool],
        "Garage_Size": [garage_size],
        "Distance_to_Center": [distance_to_center],
        "Num_Bathrooms": [0],  # Assuming default values for missing features
        "Num_Floors": [1],
        "Year_Built": [2000],  # Set as default or some other reasonable value
        "Location_Score": [0],  # Same here, adjust as needed
        "ID": [0]  # Not needed for prediction, just a placeholder
    })

    # Ensure the input data columns match the model's expected features
    input_data = input_data[model.feature_names_in_]

    # Predict price using the trained model
    try:
        predicted_price = model.predict(input_data)[0]
        st.subheader(f"Predicted Price: ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")



# Main Function
def main():
    st.title("House Price Prediction App")
    st.markdown("### Predict house prices based on your preferences!")

    # Navigation
    section = sidebar_navigation()

    # Render sections based on user selection
    if section == "EDA":
        eda_section(df)
    elif section == "Insights":
        insights_section(df)
    elif section == "Model":
        model_section(model)

# This line is crucial to run the main function and display the app content on the local server.
if __name__ == "__main__":
    main()
