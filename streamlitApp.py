import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
#load the updated model to be made live
model = joblib.load("liveModelV1.pkl")
#load the data set
data = pd.read_csv("mobile_price_range_data.csv")
X= data.iloc[:,:-1]
y= data.iloc[:, -1]
#train test split for accuracy clalculation  
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
#make predicition for X_test set
y_pred = model.predict(X_test)
#calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
#page title
st.title("model Accuracy and real_time Prediction")
#display accuracy
st.write(f"Model {accuracy}")
#real time prediction based on users input
st.header("Real_time prediction")
input_data=[]
for col in X_test.columns:
   input_value = st.number_input(f'input for feature{col}',value=0)
   input_data.append(input_value)
#convert input dataa to dataframe
input_df = pd.DataFrame([input_data],columns=X_test.columns)
#make prediction
if st.button("predict"):
   prediction = model.predict(input_df)
   st.write(f'prediction:{prediction[0]}')





