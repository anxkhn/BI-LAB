import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")


# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/anxkhn/BI-LAB/main/MPR/Placement_Data.csv"
    df = pd.read_csv(url)
    return df


# Preprocess data
def preprocess_data(df):
    df1 = df.copy()
    df1.drop(["sl_no", "ssc_b", "hsc_b"], axis=1, inplace=True)
    le = LabelEncoder()
    lst = ["gender", "hsc_s", "degree_t", "workex", "specialisation", "status"]
    for i in lst:
        df1[i] = le.fit_transform(df1[i])
    df1["salary"] = df1["salary"].fillna(df1["salary"].mode()[0])
    X = df1.iloc[:, :10]
    y = df1.iloc[:, -2]
    ms = MinMaxScaler()
    X_sc = ms.fit_transform(X)
    return X_sc, y


# Train models
def train_models(X, y):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    svc = SVC()
    svc.fit(X, y)

    nb = GaussianNB()
    nb.fit(X, y)

    dt = DecisionTreeClassifier(criterion="gini", random_state=1)
    dt.fit(X, y)

    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(X, y)

    ad = AdaBoostClassifier()
    ad.fit(X, y)

    gb = GradientBoostingClassifier()
    gb.fit(X, y)

    xgb = XGBClassifier()
    xgb.fit(X, y)

    return knn, svc, nb, dt, rf, ad, gb, xgb


# Streamlit app
def main():
    st.title("Placement Prediction App")
    st.sidebar.title("Model Selection")

    # Load data
    df = load_data()

    # Data preprocessing
    X, y = preprocess_data(df)

    # Train models
    knn, svc, nb, dt, rf, ad, gb, xgb = train_models(X, y)

    # User input for new data
    st.sidebar.subheader("Input New Data")
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    ssc_p = st.sidebar.slider("SSC Percentage", 0.0, 100.0, 50.0)
    hsc_p = st.sidebar.slider("HSC Percentage", 0.0, 100.0, 50.0)
    hsc_s = st.sidebar.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])
    degree_p = st.sidebar.slider("Degree Percentage", 0.0, 100.0, 50.0)
    degree_t = st.sidebar.selectbox("Degree Type", ["Comm&Mgmt", "Sci&Tech", "Others"])
    workex = st.sidebar.radio("Work Experience", ["Yes", "No"])
    etest_p = st.sidebar.slider("Employability Test Percentage", 0.0, 100.0, 50.0)
    specialisation = st.sidebar.selectbox("MBA Specialization", ["Mkt&HR", "Mkt&Fin"])

    # Convert user input into dataframe
    input_data = pd.DataFrame(
        {
            "gender": [1 if gender == "Male" else 0],
            "ssc_p": [ssc_p],
            "hsc_p": [hsc_p],
            "hsc_s": [0 if hsc_s == "Commerce" else (1 if hsc_s == "Science" else 2)],
            "degree_p": [degree_p],
            "degree_t": [
                0 if degree_t == "Comm&Mgmt" else (1 if degree_t == "Sci&Tech" else 2)
            ],
            "workex": [1 if workex == "Yes" else 0],
            "etest_p": [etest_p],
            "specialisation": [1 if specialisation == "Mkt&HR" else 0],
            "mba_p": [0.0],  # Placeholder for 'mba_p' as it's not provided by user
        }
    )

    # Make predictions
    st.subheader("Placement Predictions for Input Data")
    for model, clf in zip(
        [
            "K Neighbors Classifier",
            "Support Vector Classifier",
            "Naive Bayes",
            "Decision Tree",
            "Random Forest",
            "AdaBoost",
            "Gradient Boosting",
            "XGBoost",
        ],
        [knn, svc, nb, dt, rf, ad, gb, xgb],
    ):
        prediction = clf.predict(input_data)
        prediction_text = "Placed" if prediction[0] == 1 else "Not Placed"
        color = "green" if prediction[0] == 1 else "red"
        st.write(
            f'<span style="color:{color}">{model}: {prediction_text}</span>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
