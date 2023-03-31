import pickle
import joblib
import streamlit as st
import pandas as pd

onehot = joblib.load("onehotencoder")
minmax = joblib.load("minmaxscaler")
model = pickle.load(open('bestknn.pkl', 'rb'))

def knn_prediction(type, step,amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest,newbalanceDest):
    a = pd.DataFrame([[step,type,amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest,newbalanceDest]],columns=["step","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"])
    b = pd.DataFrame([[step,amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest,newbalanceDest]],columns=["step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"])
    df = onehot.transform(a)
    df2 = minmax.transform(b)
    df = pd.DataFrame(df.toarray())
    df2 = pd.DataFrame(df2)
    clean_data = pd.concat([df2, df], axis=1 , ignore_index = True)
    prediction = model.predict(clean_data)
    if prediction[0] == 0.0:
        pred = "Not Fraud"
    else:
        pred = "Fraud"
    return pred

def main():
    st.title("Fraud Detection")
    html_temp = """
     
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    type = st.selectbox('Type Of Transaction',('CASH_IN', 'CASH_OUT','DEBIT', 'PAYMENT','TRANSFER')) 
    
    step = st.number_input("Step(1-10)")
    amount = st.number_input("Amount of transaction")
    oldbalanceOrg = st.number_input("Old Balance of Original Account")
    newbalanceOrig = st.number_input("New Balance of Original Account")
    oldbalanceDest = st.number_input("Old Balance of Destination Account")
    newbalanceDest = st.number_input("New Balance of Destination Account")

    result=" "

    if st.button("Predict"):
        print(type)
        result = knn_prediction(type=type,step=step,amount=amount,oldbalanceOrg=oldbalanceOrg,newbalanceOrig=newbalanceOrig,
                                oldbalanceDest=oldbalanceDest,newbalanceDest=newbalanceDest)
    st.success(f'The transaction is : {result}')
     
if __name__=='__main__':
    main()