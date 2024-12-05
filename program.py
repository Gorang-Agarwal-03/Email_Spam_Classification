import streamlit as st
import pickle

model = pickle.load(open('model.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))

def main():
    st.title("Spam Classification Application")
    st.write("This is a Machine Learning Application")
    st.subheader("Classification")
    user_input = st.text_area("Entre the Email Here:",height=150)
    if(st.button("Predict")):
        if(user_input):
            data = [user_input]
            print(data)
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            if(result[0]==0):
                st.success("Not a Spam")
            else:
                st.error("Spam")
        else:
            st.write("Please entre an email")
main()
