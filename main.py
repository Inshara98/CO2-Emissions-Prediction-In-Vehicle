# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def add_custom_css():
    st.markdown(
        """
        <style>
        /* Use the specified colors for the background */
        .stApp {
            background: linear-gradient(#ddd6f3 → #faaca8);
            background-attachment: fixed;
            color: white;
            font-family: 'Raleway', sans-serif;
        }

        /* Sidebar customization */
        .stSidebar .css-1d391kg {
            background-color: rgba(58, 28, 113, 0.9);
            color: white;
        }

        /* Custom button styles for Predict CO2 */
        .stButton>button {
            background: linear-gradient(120deg, #7b4397, #dc2430);
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(120deg, #dc2430, #7b4397);
        }

        /* Custom input styles */
        input[type="number"] {
            border: 2px solid #b389b2;
            border-radius: 10px;
            padding: 5px;
        }

        /* Containers for decision tree and prediction */
        .stContainer {
            background: rgba(255, 255, 255, .9);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        /* Headers and titles */
        h1, h2, h3, h4 {
            color: white;
        }

        /* Styling for dataset overview tables with the specified gradient */
        .dataframe {
            background: linear-gradient(120deg, #ddd6f3, #faaca8);
            border-radius: 12px;
            padding: 10px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }

        /* Styling for the decision tree section */
        #decision-tree-container {
            background: rgba(0, 0, 0, 0.5);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            color: white;
        }

        /* Styling for the decision tree line to match background colors */
        .decision-tree-line {
            color: white;
            stroke: linear-gradient(120deg, #3a1c71, #d76d77, #ffaf7b);
            stroke-width: 2px;
        }

        /* Styling for CO2 prediction section */
        #prediction-container {
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            color: #333;
        }

        /* Font style for result display */
        #prediction-result {
            color: #b21b45; /* Changed color to blue */
            font-size: 1.5em;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True
    )

# Load the dataset
def load_data():
    path = r'c:\Users\Hp\Downloads\FuelConsumptionCo2 - FuelConsumptionCo2.csv'
    data = pd.read_csv(path)
    return data

# Apply the custom CSS
add_custom_css()

# Sidebar for navigation
st.sidebar.title("🚗 Machine Learning Project")
option = st.sidebar.selectbox('Choose a section', ['Dataset Overview', 'EDA', 'Model Building', 'Predict CO2 Emissions'])

# Main app content
st.title("💻 CO2 Emissions Prediction App")
data = load_data()

if option == 'Dataset Overview':
    st.header("Dataset Overview")
    st.write("Here is a quick look at the dataset:")
    
    # Displaying data with custom table styling
    st.markdown('<div class="dataframe">', unsafe_allow_html=True)
    st.dataframe(data.head())
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Summary statistics:")
    st.markdown('<div class="dataframe">', unsafe_allow_html=True)
    st.write(data.describe())
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Missing values in the dataset:")
    st.write(data.isnull().sum())

elif option == 'EDA':
    st.header("Exploratory Data Analysis (EDA)")
    
    st.subheader("Distribution of CO2 Emissions:")
    plt.figure(figsize=(10, 6))
    sns.histplot(data['CO2EMISSIONS'], kde=True)
    st.pyplot(plt)

    st.subheader("Scatter plot: Engine Size vs CO2 Emissions")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='ENGINESIZE', y='CO2EMISSIONS', data=data)
    st.pyplot(plt)

    # Boxplot for CO2 emissions
    st.subheader("Boxplot: CO2 Emissions vs Car Features")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='FUELTYPE', y='CO2EMISSIONS', data=data)
    st.pyplot(plt)

elif option == 'Model Building':
    # Wrapping the decision tree content in a styled container
    st.markdown('<div id="decision-tree-container">', unsafe_allow_html=True)
    st.header("Decision Tree Model")
    st.write("We will predict CO2 emissions based on engine size using a Decision Tree model.")
    
    # Model building
    X = data[['ENGINESIZE']]
    y = data['CO2EMISSIONS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Decision Tree model
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)

    # Making predictions
    y_pred = dt.predict(X_test)

    # Model Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R-squared: {r2}')

    # Plotting the Decision Tree regression results with a new line color (matching background colors)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test['ENGINESIZE'], y=y_test)
    plt.plot(X_test['ENGINESIZE'], y_pred, color='#3a1c71', label='Decision Tree Prediction')
    plt.title('Decision Tree: Engine Size vs CO2 Emissions')
    plt.legend()
    st.pyplot(plt)
    st.markdown('</div>', unsafe_allow_html=True)

elif option == 'Predict CO2 Emissions':
    # Wrapping the prediction section in a styled container
    st.markdown('<div id="prediction-container">', unsafe_allow_html=True)
    st.header("Predict CO2 Emissions Based on Engine Size")

    st.write("Use the trained Decision Tree model to predict CO2 emissions by inputting an engine size.")

    # Input for engine size
    engine_size_input = st.number_input("Enter engine size (in liters):", min_value=0.0, step=0.1)

    if engine_size_input > 0:
        # Using the previously trained Decision Tree model
        X = data[['ENGINESIZE']]
        y = data['CO2EMISSIONS']

        # Split and train the model (retraining here for the sake of continuity in the app)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X_train, y_train)

        # Make a prediction based on the user input
        predicted_co2 = dt.predict([[engine_size_input]])

        # Display the result with custom styling
        st.markdown(f'<p id="prediction-result">Predicted CO2 Emissions for engine size {engine_size_input}L: {predicted_co2[0]:.2f} g/km</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
