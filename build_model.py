import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import pickle


def load_data(filepath: str)->pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def transform_data(df:pd.DataFrame)-> pd.DataFrame:
    df['Gender']= df['Gender'].map({'Male':0, 'Female':1})
    df['Married']= df['Married'].map({'No':0, 'Yes':1})
    df['Loan_Status']= df['Loan_Status'].map({'N':0, 'Y':1})

    print('Dropping Missing Values\n')
    df = df.dropna()

    print("selecting Features..\n")
    X = df[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
    y = df.Loan_Status
    print(X.shape, y.shape)

    return X, y
    
def build_model(X:pd.DataFrame, y:pd.Series)->RandomForestClassifier:
    model = RandomForestClassifier(max_depth=4, random_state = 10) 
    model.fit(X, y)

    return model

def evaluate(model:RandomForestClassifier, X:pd.DataFrame, y:pd.Series):
    pred_cv = model.predict(X)
    print(f"Accuracy Score: {accuracy_score(y,pred_cv)*100}")

def save_model(model:RandomForestClassifier, model_name:str):
    pickle_out = open(model_name, mode = "wb") 
    pickle.dump(model, pickle_out) 
    pickle_out.close()

def main():
    # load the data
    df = load_data('data.csv')
    # transform the data
    X, y = transform_data(df)
    # build the model
    classifier = build_model(X, y)
    # get evaluation
    evaluate(classifier, X, y)
    # pickle the model
    save_model(classifier, model_name='model/loan_predictor_model.pkl')

if __name__ == "__main__":
    main()
