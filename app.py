import pickle
import streamlit as st


st.set_page_config(
    page_title="Loan Eligibility Predition", page_icon=":moneybag:", layout="wide"
)

# loading the trained model
def load_model(model_path: str):
    pickled_model = open(model_path, "rb")
    classifier = pickle.load(pickled_model)
    return classifier


# defining the function which will make the prediction using the data which the user inputs
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):

    classifier = load_model("model/loan_predictor_model.pkl")

    # Pre-processing user input
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1

    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1

    if Credit_History == "Unclear Debts":
        Credit_History = 0
    else:
        Credit_History = 1

    LoanAmount = LoanAmount / 1000

    # Making predictions
    prediction = classifier.predict(
        [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]]
    )

    if prediction == 0:
        pred = "Rejected"
    else:
        pred = "Approved"
    return pred


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """
    <div style ="background-color:#2a3990;padding:15px">
    <h2 style ="color:#ffff00;text-align:center;">Bank Loan Eligibility Prediction</h2>
    </div>
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    left_column, right_column = st.columns([0.5, 0.5])
    Gender = left_column.selectbox("Gender", ("Male", "Female"))
    Married = right_column.selectbox("Marital Status", ("Unmarried", "Married"))
    ApplicantIncome = left_column.number_input("Applicant monthly income ($)")
    LoanAmount = right_column.number_input("Total loan amount ($)")
    Credit_History = st.selectbox(
        "Credit History", ("Unclear Debts", "No Unclear Debts")
    )
    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict "):
        result = prediction(
            Gender, Married, ApplicantIncome, LoanAmount, Credit_History
        )
        if result == "Rejected":
            st.error(f"Loan is {result}\n ## Sorry ** :sob: ** ")

        else:
            st.success(
                "loan is {}\n ## Congratulations ** :thumbsup: :dollar: **".format(
                    result
                )
            )
            st.balloons()


if __name__ == "__main__":
    main()
