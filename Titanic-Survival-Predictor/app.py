import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Set the page config as the first command
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Preprocess the dataset
df.dropna(subset=['age', 'embarked', 'sex', 'pclass'], inplace=True)  # Drop rows with missing essential data
df['sex'] = LabelEncoder().fit_transform(df['sex'])  # Encode sex column (Male=1, Female=0)
df['embarked'] = LabelEncoder().fit_transform(df['embarked'])  # Encode Embarked column
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)  # Fill missing Embarked values

# Feature and target variables
features = ['pclass', 'age', 'sex', 'sibsp', 'parch', 'embarked']
X = df[features]
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar for model selection
st.sidebar.header('Model Selection')
model_choice = st.sidebar.selectbox('Choose a Model:', [
    'Random Forest', 'Logistic Regression', 'Support Vector Machine', 'K-Nearest Neighbors', 
    'Decision Tree', 'Gradient Boosting', 'AdaBoost', 'XGBoost', 'Neural Network', 'Naive Bayes'])

# Sidebar for model tuning based on model choice
if model_choice == 'Random Forest':
    n_estimators = st.sidebar.slider('Number of Estimators (Trees)', 50, 300, 100, 10)
    max_depth = st.sidebar.slider('Max Depth of Trees', 1, 20, 10)
    min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 10)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
elif model_choice == 'Logistic Regression':
    C = st.sidebar.slider('Regularization Strength (C)', 0.01, 10.0, 1.0)
    model = LogisticRegression(C=C, random_state=42)
elif model_choice == 'Support Vector Machine':
    C = st.sidebar.slider('Regularization Strength (C)', 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
elif model_choice == 'K-Nearest Neighbors':
    n_neighbors = st.sidebar.slider('Number of Neighbors (n_neighbors)', 1, 20, 5)
    weights = st.sidebar.selectbox('Weighting Method', ['uniform', 'distance'])
    metric = st.sidebar.selectbox('Distance Metric', ['euclidean', 'manhattan', 'minkowski'])
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
elif model_choice == 'Decision Tree':
    max_depth = st.sidebar.slider('Max Depth', 1, 20, 10)
    min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 10)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
elif model_choice == 'Gradient Boosting':
    n_estimators = st.sidebar.slider('Number of Estimators', 50, 300, 100, 10)
    learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.1)
    max_depth = st.sidebar.slider('Max Depth', 1, 20, 10)
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
elif model_choice == 'AdaBoost':
    n_estimators = st.sidebar.slider('Number of Estimators', 50, 300, 100, 10)
    learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.1)
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
elif model_choice == 'XGBoost':
    n_estimators = st.sidebar.slider('Number of Estimators', 50, 300, 100, 10)
    learning_rate = st.sidebar.slider('Learning Rate', 0.01, 1.0, 0.1)
    max_depth = st.sidebar.slider('Max Depth', 1, 20, 10)
    model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
elif model_choice == 'Neural Network':
    hidden_layer_sizes = st.sidebar.slider('Hidden Layer Sizes', 10, 100, 50)
    activation = st.sidebar.selectbox('Activation Function', ['relu', 'tanh', 'logistic'])
    model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), activation=activation, random_state=42)
else:  # Naive Bayes
    model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Get feature importance for models that support it
if hasattr(model, 'feature_importances_'):
    feature_importance = model.feature_importances_
else:
    feature_importance = np.zeros_like(features)

# Streamlit app content
st.title("Titanic Survival Prediction")
st.markdown("### Welcome to the Titanic Survival Prediction App")
st.markdown("""
    This app predicts the likelihood of a passenger surviving the Titanic disaster based on various features such as age, 
    passenger class, sex, and more. 
    The model is built using your selected model, and you can input your own data to see the survival prediction.
    
    ### Data Description
    Below is an explanation of the key features in the Titanic dataset used in this model:
    
    - **Pclass**: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
    - **Age**: Age of the passenger in years.
    - **Sex**: Gender of the passenger (0 = Female, 1 = Male).
    - **SibSp**: Number of siblings or spouses aboard.
    - **Parch**: Number of parents or children aboard.
    - **Embarked**: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).
    - **Survived**: Whether the passenger survived (0 = No, 1 = Yes).
""")

# User input fields for prediction
st.header("Input Passenger Data")
pclass = st.selectbox('Pclass (Passenger Class)', [1, 2, 3], index=0)
age = st.slider('Age', 1, 100, 30)
sex = st.selectbox('Sex', ['Male', 'Female'])
sibsp = st.slider('SibSp (Number of Siblings/Spouse aboard)', 0, 10, 1)
parch = st.slider('Parch (Number of Parents/Children aboard)', 0, 10, 0)
embarked = st.selectbox('Embarked (Port of Embarkation)', ['C', 'Q', 'S'])

# Encode the user inputs
sex = 1 if sex == 'Male' else 0
embarked = {'C': 0, 'Q': 1, 'S': 2}[embarked]

# Create a DataFrame for the user input
user_input = pd.DataFrame([[pclass, age, sex, sibsp, parch, embarked]],
                          columns=features)

# Prediction
prediction = model.predict(user_input)[0]
prediction_prob = model.predict_proba(user_input)[0][prediction]

# Display prediction
st.subheader('Survival Prediction')
if prediction == 1:
    st.write(f"**Survived**: Yes (with {prediction_prob*100:.2f}% probability)")
else:
    st.write(f"**Survived**: No (with {(1-prediction_prob)*100:.2f}% probability)")

# Display model accuracy
st.subheader('Model Accuracy')
st.write(f"The accuracy of the model is: {accuracy*100:.2f}%")

# Display feature importance (only for models that support it)
if hasattr(model, 'feature_importances_'):
    st.subheader('Feature Importance')
    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_df, ax=ax, palette="Blues_d")
    ax.set_title('Feature Importance')
    st.pyplot(fig)

# Confusion Matrix
st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)

# ROC Curve
st.subheader('Receiver Operating Characteristic (ROC) Curve')
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
st.pyplot(fig)

# Add a footer for credits
st.markdown("---")
st.markdown("### Built by Oliver Carey")
st.markdown("[GitHub](https://github.com/olivercareyncl) | [LinkedIn](https://www.linkedin.com/in/oliver-carey/)")

