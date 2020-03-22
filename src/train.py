from sklearn import preprocessing
from sklearn import ensemble
import pandas as pd
import os
from sklearn import metrics
from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
TEST_DATA = os.environ.get("TEST_DATA")

print("Fold file path: " + TRAINING_DATA)
print("Fold num: " + str(FOLD))
print("MODEL Used: " + MODEL)
print("MOTEST DATADEL Used: " + TEST_DATA)

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}


if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    print(df.head())
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis = 1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis = 1)


    label_encoders = {}

    for col in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[col].values.tolist() + valid_df[col].values.tolist() + df_test[col].values.tolist())
        train_df.loc[:, col] = lbl.transform(train_df[col].values.tolist())
        valid_df.loc[:, col] = lbl.transform(valid_df[col].values.tolist())

        label_encoders[col] = lbl


    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(preds)
    print(metrics.roc_auc_score(yvalid, preds))


    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")













