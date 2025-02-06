import pickle
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

from sklearn.preprocessing import StandardScaler  

st.set_page_config(
    page_title="Prediction of Disease Outbreaks",
    layout="wide",
    page_icon="🧑‍⚕️",
)

# Load the models
diabetes_model = pickle.load(open(r"C:\Users\HP\Desktop\PDO1\training_model\diabetes_model.sav" ,'rb'))
heart_model=pickle.load(open(r"C:\Users\HP\Desktop\PDO1\training_model\heart_model.sav",'rb'))
parkinson_model=pickle.load(open(r"C:\Users\HP\Desktop\PDO1\training_model\parkinson_model.sav",'rb'))

scaler_diabetes = pickle.load(open(r"C:\Users\HP\Desktop\PDO1\training_model\scaler.sav", 'rb'))
scaler_heart=pickle.load(open(r"C:\Users\HP\Desktop\PDO1\training_model\scaler1.sav",'rb'))
scaler=pickle.load(open(r"C:\Users\HP\Desktop\PDO1\training_model\scaler2.sav","rb"))
# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        "Prediction of Disease Outbreaks",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Disease Prediction"],
        menu_icon="hospital-fill",
        icons=["activity", "heart", "person"],
        default_index=0,
    )

# Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction ")

    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")

    with col1:
        SkinThickness = st.text_input("Skin Thickness Value")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        BMI = st.text_input("BMI Value")

    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    with col2:
        Age = st.text_input("Age")

    # Prediction
    if st.button("Predict Diabetes"):
        if (
            Pregnancies
            and Glucose
            and BloodPressure
            and SkinThickness
            and Insulin
            and BMI
            and DiabetesPedigreeFunction
            and Age
        ):
            # Convert inputs to float
            input_data = np.array([
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age),
            ]).reshape(1, -1)

            # Apply scaling 
            scaled_input = scaler_diabetes.transform(input_data)

            #  prediction
            dia_prediction =diabetes_model.predict(scaled_input)

            

            if dia_prediction[0] == 1:
                st.success("The person is likely to have diabetes.")
            else:
                st.success("The person is not likely to have diabetes.")
        else:
            st.warning("Please fill all the fields before predicting.")
#heart disease page
elif selected =="Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    #input fields
    col1,col2,col3=st.columns(3)


    with col1:
        age=st.text_input("Age")
    with col2:
        sex=st.text_input("sex(1=Male,0=Female) ")
    with col3:
        cp=st.text_input("chest pain type(0-3)")

    with col1:
        trestbps=st.text_input("Resting Blood Pressure(mm Hg)")
    with col2:
             chol  =st.text_input("Serum  Cholesterol(mg/dL)")
    with col3:
       fbs =st.text_input("Fasting Blood Sugar(>120 mg/dl,1=True,0=False)")

    with col1:
       restecg = st.text_input("Resting ECG Results (0-2)")
    with col2:
      thalach =st.text_input("Maximum Heart Rate Achieved")
    with col3:
        exang=st.text_input("Exercise-Induced Angina (1 = Yes, 0 = No")
       
    with col1:
       oldpeak=st.text_input("ST Depression Induced by Exercise")
    with col2:
       slope=st.text_input("Slope of the Peak Exercise ST Segment (0-2)")
    with col3:
        ca= st.text_input("Number of Major Vessels (0-4)")
 
    with col1:
       thal =st.text_input("Thalassemia Type (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)")
    
      
     #Prediction
    if st.button("Predict Heart Disease"):
        if(
            age
            and sex
            and cp
            and trestbps
            and chol
            and fbs
            and restecg
            and thalach
            and exang
            and oldpeak
            and slope
            and ca
            and thal
        ):   #Convert inputs to float
            input_data_heart=np.array([
             float(age),
             float(sex),
             float(cp),
             float(trestbps),
             float(chol),
             float(fbs),
             float(restecg),
             float(thalach),
             float(exang),
             float(oldpeak),
             float(slope),
             float(ca),
             float(thal),
            
            ]).reshape(1,-1)

            #Apply Scaling
            scaled_inputs_heart = scaler_heart.transform(input_data_heart)

            #Make prediction
            heart_prediction=heart_model.predict(scaled_inputs_heart)
            

            if  heart_prediction[0]==1:
                st.success("The person is likely to have Heart Disease.")
            else:
                st.success("The person is not likely to have Heart Disease.")
        else:
            st.warning("Please fill all the fields before predicting.")


# Parkinson Prediction Page
                  
elif selected=="Parkinson's Disease Prediction":
    st.title("Parkinson Disease")
    #Input fields
    col1,col2,col3=st.columns(3)

    with col1:
        MDVP_Fo_Hz=st.text_input("MDVP Fo(Hz)")
    with col2:
        MDVP_Fhi_Hz=st.text_input("MDVP Fhi(Hz)")
    with col3:
        MDVP_Flo_Hz=st.text_input("MDVP Flo(Hz)")

    with col1:
           
        MDVP_Jitter_percent= st.text_input("MDVP Jitter(%)")
    with col2:
        MDVP_Jitter_Abs =st.text_input("MDVP Jitter(Abs)")
    with col3:
        MDVP_RAP=st.text_input("MDVP RAP")
    
    with col1:
       MDVP_PPQ =st.text_input("MDVP PPQ")
    with col2:
        Jitter_DDP=st.text_input("Jitter:DDP")
    with col3:
        MDVP_Shimmer=st.text_input("MDVP Shimmer")

    with col1:
        MDVP_Shimmer_dB=st.text_input("MDVP Shimmer(dB)")
    with col2:
       Shimmer_APQ3 =st.text_input("Shimmer APQ3")
    with col3:
            Shimmer_APQ5=st.text_input("Shimmer APQ5")
    
    with col1:
        MDVP_APQ=st.text_input("MDVP APQ")
    with col2:
       Shimmer_DDA =st.text_input("Shimmer DDA")
    with col3:
        NHR =st.text_input("NHR")
    
    with col1:
        HNR=st.text_input("HNR")
    with col2:
        RPDE=st.text_input("RPDE")
    with col3:
        DFA=st.text_input("DFA")
    
    with col1:
        spread1=st.text_input("spread1")
    with col2:
         spread2=st.text_input("spread2")
    with col3:
        D2=st.text_input("D2")
    with col1:
      PPE  =st.text_input("PPE")
    #prediction
    if st.button("Predict Parkinson Disease"):
       if(
            
             MDVP_Fo_Hz
            and MDVP_Fhi_Hz
            and MDVP_Flo_Hz
            and MDVP_Jitter_percent
            and MDVP_Jitter_Abs
            and MDVP_RAP
            and MDVP_PPQ
            and Jitter_DDP
            and MDVP_Shimmer
            and MDVP_Shimmer_dB
            and Shimmer_APQ3
            and Shimmer_APQ5
            and MDVP_APQ
            and Shimmer_DDA
            and NHR 
            and HNR
            and RPDE
            and DFA
            and spread1
            and spread2
            and D2
            and PPE
       ):  #convert ibputs to float
           inputs_data_parki=np.array([
             float( MDVP_Fo_Hz), 
             float(MDVP_Fhi_Hz),
            float(MDVP_Flo_Hz) ,
            float (MDVP_Jitter_percent),
            float( MDVP_Jitter_Abs),
            float(MDVP_RAP),
             float(MDVP_PPQ),
             float(Jitter_DDP),
             float(MDVP_Shimmer),
             float(MDVP_Shimmer_dB),
             float(Shimmer_APQ3),
             float(Shimmer_APQ5),
            float(MDVP_APQ),
             float(Shimmer_DDA),
            float(NHR),  
            float(HNR),
            float(RPDE),
           float(DFA), 
            float(spread1),
             float(spread2),
             float(D2),
            float(PPE ), 
            ]).reshape(1,-1)
           
           # Apply scaling 
       # Print the type to confirm it's a scaler
           scaled_inputs_park = scaler.transform(inputs_data_parki)  # Then use it to transform the data
           pred_parki=parkinson_model.predict(scaled_inputs_park)

           

           if  pred_parki[0]==1:
                st.success("The person is likely to have Parkinsons Disease. ")
           else:
                st.success("The person is not likely to have Parkinsons Disease. ")
       else:
            st.warning("Please fill all the fields before predicting")