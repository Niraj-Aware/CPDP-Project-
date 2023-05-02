import streamlit as st

# Define the function for the Farmer button
def farmer():
    st.markdown("Redirecting to the Farmer app...")
    st.experimental_set_query_params(option="farmer")
    st.experimental_redirect("https://niraj-aware-cpdp-project--cpdpapp-vpt5d7.streamlit.app/")

# Define the function for the Pharmacist button
def pharmacist():
    st.markdown("Redirecting to the Pharmacist app...")
    st.experimental_set_query_params(option="pharmacist")
    st.experimental_redirect("https://niraj-aware-cpdp-project--app-y9w0bf.streamlit.app/")

# Define the Streamlit app
def main():
    st.title("Welcome to the Farmer and Pharmacist app!")
    st.write("Please select an option below:")

    # Add a button for the Farmer option
    if st.button("Farmer"):
        farmer()

    # Add a button for the Pharmacist option
    if st.button("Pharmacist"):
        pharmacist()

if __name__ == "__main__":
    main()



