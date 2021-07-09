import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
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

def explained_variance(model:RandomForestClassifier,X:pd.DataFrame, y:pd.Series)-> float:
    train_score = model.score(X, y) * 100

    return train_score

def report_accuracy(model:RandomForestClassifier, X:pd.DataFrame, y:pd.Series)->float:
    train_pred = model.predict(X)
    train_accuracy = accuracy_score(train_pred, y) * 100

    return train_accuracy


def write_metrics(filename:str, train_score:float, train_accuracy:float ):
    with open(filename, "w") as out_file:
        out_file.write("Training Variance Explained:: %2.1f%%\n"% train_score)

        out_file.write("Training Accuracy Score:: %2.1f%%\n"% train_accuracy)


def main():
    # load the data
    df = load_data('data/data.csv')
    # transform the data
    X, y = transform_data(df)
    # build the model
    classifier = build_model(X, y)
    # get evaluation
    evaluate(classifier, X, y)
    # pickle the model
    save_model(classifier, model_name='model/loan_predictor_model.pkl')
    # explained variance
    train_score = explained_variance(classifier, X, y)
    # train_accuracy
    train_accuracy = report_accuracy(classifier, X, y)
    # write metrics
    write_metrics("metrics.txt", train_score, train_accuracy)

    ### Plot Feature Importance
    feature_importance = classifier.feature_importances_
    labels = X.columns
    feature_df = pd.DataFrame(list(zip(labels, feature_importance )), columns=['Features', 'Importance'])
    feature_df = feature_df.sort_values(by="Importance", ascending=False)

    ## Image Formatting
    axis_fs = 18
    title_fs = 22
    sns.set_style("whitegrid")
    plt.figure(figsize=(15,10))
    ax = sns.barplot(x="Importance", y='Features', data=feature_df)
    ax.set_xlabel("Importance", fontsize=axis_fs)
    ax.set_ylabel("Features", fontsize=axis_fs)
    ax.set_title("Random Forest Classifier\n Feature Importance", fontsize=title_fs)

    plt.tight_layout()
    plt.savefig('featureImportance.png', dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
