
![docker image deployed](https://github.com/Abuton/LoanPredictor/actions/workflows/deploy_docker.yml/badge.svg)
![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![githubAction](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)
![streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

# Loan Eligibility Predictor

A Machine Learning Model to predict Loan Eligibility for Dream Housing Finance company

Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.

Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers.

## Loan Eligibity Prediction App

The App was Built to Expose the Model into production -  a stage called deployement in Machine Learning
The following Steps were taking

- Model was Built using the Random Forest Classifier
- The model was then pickled using the pickle library
- A frontend view was designed to collect input data using streamlit
- A Web App was deployed using Streamlit Sharing and heroku

Data Source [here](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/?utm_source=blog&utm_medium=model_depoyment_using_streamlit#ProblemStatement)

## How to use the App

![image](https://user-images.githubusercontent.com/40719064/125074167-d491eb80-e0b4-11eb-8021-97d465070e33.png)

The Above Image is the front end design of the App. It has the following fields

1. Gender - A select box to choose the gender of the loan applicant
2. Monthly Income - An Input box to enter the monthly income of the applicant
3. Credit History - A select box to chose if the applicant has unclear debt or not
4. Marital Status - A select box to choose if the applicant is married
5. Loan Amount - A number input inbox to enter the amount an applicant wants to borrowed
