import streamlit as st
import pandas as pd
import joblib
import os
from streamlit_option_menu import option_menu

# get the directory/path of the saved model (absolute path)

working_dir = os.path.dirname(os.path.abspath(__file__))

# print(working_dir)

#Penguin Model:opt_SVM_model

# load the saved model

Penguin_model = joblib.load(os.path.join(working_dir, "opt_SVM_model"))
print(Penguin_model)

# sidebar for navigation

with st.sidebar:
    selected = option_menu("Multiple Classifications System",
                           ['Penguin species prediction',
                            'Iris species prediction',
                            'Diabetes prediction',
                            ],
                           menu_icon='species-fill',
                           icons=['emoji-grin', 'apple', 'person'],
                           default_index=0)
#1 Penguin species prediction
if selected == 'Penguin species prediction':
    # page title
    st.title("Penguin species prediction using SVM")

# get input
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        island = st.selectbox("Island selection", ["Biscoe", "Dream", "Torgersen"])

    with col2:
        bill_length_mm = st.number_input("Enter bill length", format="%.1f", value=30.0, min_value=30.0, max_value=60.0,
                                         step=0.1)

    with col3:
        bill_depth_mm = st.number_input("Enter bill depth", format="%.1f", value=12.0, min_value=12.0, max_value=20.0,
                                        step=0.1)

    with col4:
        flipper_length_mm = st.number_input("Enter flipper length", format="%d", value=150, min_value=150,
                                            max_value=300,
                                            step=1)

    with col5:
        body_mass = st.number_input("Enter body mass", format="%d", value=3000, min_value=3000,
                                    max_value=6000,
                                    step=1)

    with col6:
        sex = st.selectbox("Gender", ["Male", "Female"])

    prediction_btn_Penguin = st.button("Prediction(Penguin)", key="predict_Penguin")
    if prediction_btn_Penguin:
        keys = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
        user_input  = [island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass, sex]
        input_dict = dict(zip(keys, user_input))
        spec_classification = Penguin_model.predict(pd.DataFrame(input_dict, index=[0]))[0]
        st.success(spec_classification)




#2 Iris species prediction
#Iris Model:iris_SVM_model
from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
X, y = iris.data, iris.target
model_iris = SVC()
model_iris.fit(X, y)
joblib.dump(model_iris, "iris_SVM_model")

# load the saved model
iris_model = joblib.load(os.path.join(working_dir, "iris_SVM_model"))
print(iris_model)

if selected == 'Iris species prediction':
    # page title
    st.title("Iris species prediction using SVM")

# get input
    col1, col2, col3, col4= st.columns(4)

    with col1:
        SepalLength_cm = st.number_input("Enter SepalLength", format="%.1f", value=5.1, min_value=4.3, max_value=7.9,
                                         step=0.1)

    with col2:
        SepalWidth_cm = st.number_input("SepalWidth", format="%.1f", value=3.5, min_value=2.0, max_value=4.4,
                                        step=0.1)

    with col3:
        PetalLength_cm = st.number_input("Enter PetalLength", format="%.1f", value=1.4, min_value=1.0,
                                            max_value=6.9, step=0.1)

    with col4:
        PetalWidth_cm = st.number_input("PetalWidth", format="%.1f", value=0.2, min_value=0.1,
                                    max_value=2.5, step=0.1)


    prediction_btn_Iris = st.button("Prediction(Iris)", key="predict_Iris")
    if prediction_btn_Iris:
        keys = ["SepalLength_cm", "SepalWidth_cm", "PetalLength_cm", "PetalWidth_cm"]
        user_input  = [SepalLength_cm, SepalWidth_cm, PetalLength_cm, PetalWidth_cm]
        input_dict = dict(zip(keys, user_input))
        pred_class = iris_model.predict(pd.DataFrame(input_dict, index=[0]))[0]
        species_name = iris.target_names[pred_class]
        st.success(species_name)


#2 Diabetes prediction
#Diabetes Model:Diabetes_SVM_model
df = pd.read_csv(os.path.join(working_dir, "diabetes - diabetes.csv"))
X = df[["Glucose", "BloodPressure", "BMI"]]   # only 3 features
y = df["Outcome"]

diabetes_model = SVC()
diabetes_model.fit(X, y)

# save the model
joblib.dump(diabetes_model, os.path.join(working_dir, "diabetes_SVM_model.pkl"))

# load the saved model
diabetes_model = joblib.load(os.path.join(working_dir, "diabetes_SVM_model.pkl"))

if selected == 'Diabetes prediction':
    # page title
    st.title("Diabetes prediction using SVM")

# get input
    col1, col2, col3= st.columns(3)

    with col1:
        Glucose = st.number_input("Glucose", format="%.d", value=148, min_value=0, max_value=199,
                                         step=0)

    with col2:
        BloodPressure = st.number_input("BloodPressure", format="%.d", value=72, min_value=0, max_value=122,
                                        step=0)

    with col3:
        BMI = st.number_input("BMI", format="%.1f", value=33.6, min_value=0.0,
                                            max_value=67.1, step=0.1)



    prediction_btn_Diabetes = st.button("Prediction(Diabetes)", key="predict_Diabetes")
    if prediction_btn_Diabetes:
        keys = ["Glucose", "BloodPressure", "BMI"]
        user_input  = [Glucose, BloodPressure, BMI]
        input_dict = dict(zip(keys, user_input))
        pred_class = diabetes_model.predict(pd.DataFrame(input_dict, index=[0]))[0]
        result = "Diabetes detected" if pred_class == 1 else "No diabetes"
        st.success(result)
