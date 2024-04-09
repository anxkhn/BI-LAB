import streamlit as st
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
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
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset
url = "https://raw.githubusercontent.com/anxkhn/BI-LAB/main/MPR/Placement_Data.csv"
df = pd.read_csv(url)

# Data preprocessing
df.drop(["sl_no", "ssc_b", "hsc_b"], axis=1, inplace=True)
le = LabelEncoder()
lst = ["gender", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for i in lst:
    df[i] = le.fit_transform(df[i])
df["salary"] = df["salary"].fillna(df["salary"].mode()[0])
X = df.iloc[:, :10]
y = df.iloc[:, -2]
ms = MinMaxScaler()
X_sc = ms.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, random_state=1
)

# Oversampling
sm = RandomOverSampler()
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Train models
models = {
    "K Neighbors Classifier": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Classifier": SVC(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(criterion="gini", random_state=1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XG Boost": XGBClassifier(),
}

for model_name, model in models.items():
    model.fit(X_train_sm, y_train_sm)

# Streamlit app
st.title("Placement Prediction App")

# User input
st.sidebar.subheader("User Input")
gender = st.sidebar.radio("Gender", ["Male", "Female"])
ssc_p = st.sidebar.slider("SSC percentage", min_value=0.0, max_value=100.0, value=0.0)
hsc_p = st.sidebar.slider("HSC percentage", min_value=0.0, max_value=100.0, value=0.0)
hsc_s = st.sidebar.selectbox("HSC specialization", ["Arts", "Commerce", "Science"])
degree_p = st.sidebar.slider(
    "Degree percentage", min_value=0.0, max_value=100.0, value=0.0
)
degree_t = st.sidebar.selectbox(
    "Degree type", ["Others", "Sci&Tech", "Comm&Mgmt", "Arts"]
)
workex = st.sidebar.selectbox("Work experience", ["No", "Yes"])
etest_p = st.sidebar.slider(
    "Employability test percentage", min_value=0.0, max_value=100.0, value=0.0
)
specialisation = st.sidebar.selectbox("Specialization", ["Mkt&Fin", "HR", "Others"])
mba_p = st.sidebar.slider("MBA percentage", min_value=0.0, max_value=100.0, value=0.0)

# Preprocess user input
gender = 1 if gender == "Female" else 0
hsc_s = ["Arts", "Commerce", "Science"].index(hsc_s)
degree_t = ["Others", "Sci&Tech", "Comm&Mgmt", "Arts"].index(degree_t)
workex = 1 if workex == "Yes" else 0
specialisation = ["Mkt&Fin", "HR", "Others"].index(specialisation)

# Create input dataframe
input_df = pd.DataFrame(
    {
        "gender": [gender],
        "ssc_p": [ssc_p],
        "hsc_p": [hsc_p],
        "hsc_s": [hsc_s],
        "degree_p": [degree_p],
        "degree_t": [degree_t],
        "workex": [workex],
        "etest_p": [etest_p],
        "specialisation": [specialisation],
        "mba_p": [mba_p],
    }
)

# Encode categorical columns
input_df["gender"] = le.fit_transform(input_df["gender"])
input_df["hsc_s"] = le.fit_transform(input_df["hsc_s"])
input_df["degree_t"] = le.fit_transform(input_df["degree_t"])
input_df["specialisation"] = le.fit_transform(input_df["specialisation"])

# Scale the input data
input_scaled = ms.transform(input_df)

# Predict
st.subheader("Prediction Results:")
for model_name, model in models.items():
    prediction = model.predict(input_scaled)
    st.write(f"{model_name}: {'Placed' if prediction[0] == 1 else 'Not Placed'}")
