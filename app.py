import streamlit as st
import numpy as np
import pickle

# Load trained model
model_path = "satisfaction_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Airline Passenger Satisfaction Prediction ✈️")

st.markdown("### Enter Passenger Details & Service Ratings")

# User Inputs
Type_of_Travel = st.selectbox("Type of Travel", ["Personal Travel", "Business Travel"])
Inflight_wifi_service = st.slider("Inflight WiFi Service (0-5)", 0, 5, 3)
Online_boarding = st.slider("Online Boarding (0-5)", 0, 5, 3)
Seat_comfort = st.slider("Seat Comfort (0-5)", 0, 5, 3)
Flight_Distance = st.number_input("Flight Distance", min_value=0, value=500)
Inflight_entertainment = st.slider("Inflight Entertainment (0-5)", 0, 5, 3)
On_board_service = st.slider("On-board Service (0-5)", 0, 5, 3)
Leg_room_service = st.slider("Leg Room Service (0-5)", 0, 5, 3)
Cleanliness = st.slider("Cleanliness (0-5)", 0, 5, 3)
Checkin_service = st.slider("Check-in Service (0-5)", 0, 5, 3)
Inflight_service = st.slider("Inflight Service (0-5)", 0, 5, 3)
Baggage_handling = st.slider("Baggage Handling (0-5)", 0, 5, 3)

# Convert categorical values to numerical
Type_of_Travel = 1 if Type_of_Travel == "Business Travel" else 0

# Prepare input data for prediction
input_data = np.array([
    Type_of_Travel, Inflight_wifi_service, Online_boarding, Seat_comfort, Flight_Distance,
    Inflight_entertainment, On_board_service, Leg_room_service, Cleanliness,
    Checkin_service, Inflight_service, Baggage_handling
]).reshape(1, -1)

# Predict Button
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_data)
    result = "Satisfied 😃" if prediction[0] == 1 else "Neutral or Dissatisfied 😐"
    
    # Display Result
    st.subheader("Prediction Result:")
    st.write(f"**{result}**")

    # Show Probability (if supported by model)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0]
        st.write(f"Confidence: **{probability.max()*100:.2f}%**")

