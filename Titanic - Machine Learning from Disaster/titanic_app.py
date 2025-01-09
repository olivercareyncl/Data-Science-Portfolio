import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Function to load the datasets
@st.cache
def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    gender_submission_df = pd.read_csv('gender_submission.csv')
    return train_df, test_df, gender_submission_df

# Display Data Exploration Info
def display_data_exploration(df):
    st.subheader('Data Overview')
    st.write(df.head())
    st.write('Summary Statistics')
    st.write(df.describe())
    st.write('Missing Values')
    st.write(df.isnull().sum())

# Display visualizations
def display_visualizations(df):
    st.subheader('Survival Rate by Gender')
    sns.barplot(x='Sex', y='Survived', data=df)
    st.pyplot()
    
    st.subheader('Survival Rate by Class')
    sns.barplot(x='Pclass', y='Survived', data=df)
    st.pyplot()
    
    st.subheader('Age Distribution')
    sns.histplot(df['Age'], kde=True, bins=30)
    st.pyplot()

# Model Building and Prediction
def build_model(df):
    # Feature Engineering
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Handle missing values for 'Age' and 'Fare'
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Fare'] = imputer.fit_transform(df[['Fare']])
    
    X = df[['Pclass', 'Age', 'Fare', 'Sex']]
    y = df['Survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression Model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# User Input for Prediction
def user_input_for_prediction(model):
    st.subheader("Predict Survival for a New Passenger")
    pclass = st.selectbox('Pclass', [1, 2, 3])
    age = st.slider('Age', 0, 100, 30)
    fare = st.slider('Fare', 0, 500, 50)
    sex = st.selectbox('Sex', ['male', 'female'])

    # Prepare input data for prediction
    input_data = pd.DataFrame([[pclass, age, fare, sex]], columns=['Pclass', 'Age', 'Fare', 'Sex'])
    input_data['Sex'] = input_data['Sex'].map({'male': 0, 'female': 1})
    
    # Predict survival
    prediction = model.predict(input_data)[0]
    st.write("Predicted Survival:", "Survived" if prediction == 1 else "Not Survived")

def main():
    st.title("Titanic Survival Prediction")
    
    # Load the data
    train_df, test_df, gender_submission_df = load_data()
    
    # Show data exploration and visualizations
    display_data_exploration(train_df)
    display_visualizations(train_df)
    
    # Build the model
    model, accuracy = build_model(train_df)
    
    # Show model accuracy
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    # Predict for a new passenger
    user_input_for_prediction(model)

if __name__ == "__main__":
    main()
