import pandas as pd
import numpy as np
# from sklearn import svm
import pickle
import joblib

df = pd.read_csv('./test.csv')


# make the categorical columns into one hot encoding
categorical_cols = ['Gender','H_Cal_Consump','Alcohol_Consump','Smoking','Food_Between_Meals','Fam_Hist','H_Cal_Burn','Transport']
def preprocessing(df):
    for categorical in categorical_cols:
        encoded_cat = pd.get_dummies(df[categorical], prefix=categorical, prefix_sep='_')
        df = (df.drop([categorical], axis=1)).join(encoded_cat)
    return df
df = preprocessing(df)

# load the model from disk
# with open('./model.sav', 'rb') as file:
#     loaded_model = pickle.load(file)

# load the model from disk
loaded_model = joblib.load("model.pkl")

# Make predictions on the test set
y_pred = loaded_model.predict(df)

# map the predictions to the corresponding body level like this '0' -> 'Body Level 1', '1' -> 'Body Level 2', '2' -> 'Body Level 3', '3' -> 'Body Level 4'
y_pred = np.array(['Body Level ' + str(int(i)+1) for i in y_pred])

# print(f'y_pred: {y_pred}')

# save the predictions in a new text file
np.savetxt('./preds.txt', y_pred, fmt='%s')

# calculate the accuracy of the model, confusion matrix and classification report
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# y_true = pd.read_csv('../Data/body_level_classification_train.csv')
# y_true = y_true['Body_Level']
# print(f'accuracy: {accuracy_score(y_true, y_pred)}')
# print(f'confusion matrix:\n {confusion_matrix(y_true, y_pred)}\n')
# print(f'classification report:\n {classification_report(y_true, y_pred)}')