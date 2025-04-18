
#Install dependencies:
# pip install pandas scikit-learn tensorflow streamlit

#Launch the app:
# streamlit run streamlit_app.py


import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# Load the saved model and preprocessor
@st.cache_resource
def load_model(selected_model='ANN'):

    model = tf.keras.models.load_model('tf_profit_model.h5')
    
    if selected_model=='ML':
        model = pickle.load(open('sk_profit_model.pkl', 'rb'))
        
    with open('preprocessor.pkl', 'rb') as f:
        preproc = pickle.load(f)
    return model, preproc



# App title
st.title('Startup Profit Prediction')


# Instructions

st.write("**Usage:**")
st.write("1. Ensure `50_Startups.csv` is in the same folder.\n2. Run `python train_ann_model.py` and `python train_ml_model.py` to train and save the models.\n3. Run `streamlit run streamlit_app.py` to launch the app.")
st.markdown("---")



st.write('Enter your company data below to predict the expected Profit.')


# User inputs

selected_model = st.selectbox('Select Preferred Predictor Model', options=['ANN', 'ML'])

model, preprocessor = load_model(selected_model)

rd_spend = st.number_input('R&D Spend', min_value=0.0, value=72000.0, step=1000.0)
admin = st.number_input('Administration Spend', min_value=0.0, value=80000.0, step=1000.0)
mkt_spend = st.number_input('Marketing Spend', min_value=0.0, value=3000.0, step=500.0)


# For State, you can pre-populate known categories or allow free text
state = st.selectbox('State', options=['California', 'New York', 'Florida', 'Texas', 'Other'])
if state == 'Other':
    state = st.text_input('Enter state name', '')

# Predict button
def predict_profit():
    # Build DataFrame
    df_input = pd.DataFrame([[rd_spend, admin, mkt_spend, state]],
                             columns=['R&D Spend', 'Administration', 'Marketing Spend', 'State'])
    
    # Preprocess and predict

    # ANN Model
    X_proc = preprocessor.transform(df_input)
    pred = model.predict(X_proc)
   
    if selected_model=='ML':
         return pred[0]
    else:
         return pred[0][0]


    # ML Model
    # stateEncoded = ohe.transform(np.array([[state]]))
    # finalFeatures = np.concatenate((stateEncoded,np.array([[rdSpend,admSpend,markSpend]])) , axis = 1)
    # prediction = model.predict(finalFeatures)
    # return pred[0]

if st.button('Predict Profit'):
    profit = predict_profit()
    st.success(f'Predicted Profit: ${profit:,.2f}')

