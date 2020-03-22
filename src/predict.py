from sklearn import preprocessing
from sklearn import ensemble
import pandas as pd
import os
from sklearn import metrics
from . import dispatcher
import joblib
import numpy as np

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

print("Fold file path: " + TRAINING_DATA)

print("MODEL Used: " + MODEL)



def predict():

    df = pd.read_csv(TEST_DATA)
    print(df.head())
    test_idx = df["id"].values
    predictions = None
    

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)

        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))

        for col in cols:
            lbl = encoders[col]
            df.loc[:, col] = lbl.transform(df[col].values.tolist())
            

            encoders[col] = lbl


        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]
    
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns = ["id", "target"])
    return sub



if __name__ == "__main__":

    submission = predict()
    submission.id = submission.id.astype(int)
    submission.to_csv(f"models/{MODEL}.csv", index=False)