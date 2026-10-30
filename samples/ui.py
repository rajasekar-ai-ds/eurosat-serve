import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.title("EuroSAT Land-Use Classifier")
st.caption("Upload a satellite image; the FastAPI service predicts its class. ")

uploaded = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    st.image(uploaded, caption=uploaded.name, width=256)

    if st.button("Predict"):
        files = {"file" : (uploaded.name, uploaded.getvalue(), uploaded.type)}
        try:
            with st.spinner("Asking the model..."):
                resp = requests.post(API_URL, files=files, timeout=10)

        except requests.exceptions.ConnectionError:
            st.error("Can't reach the API. Is uvicorn running on port 8000?")

        except requests.exceptions.Timeout:
            st.error("The API took too long to respond")

        else:
            if resp.status_code == 200:
                result = resp.json()
                st.success(f"Prediction: {result['class_name']}")
                st.metric("Confidence", f"{result['confidence']:.2%}")

            else:
                st.error(f"Error {resp.status_code} : {resp.json().get('detail')}")
