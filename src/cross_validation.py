
"""
Binary classification

Multi class classification

Multi label classification

Single col regression

Multi col regression

Holdout based

Homework: split regression column in such a way that distribution remains same in each fold
Holdout to be used mainly for timeseries as looking into future would give good scoes but overfit due to lekage
For timeseries and holdout don't shuffle to avoid lekage
"""

import pandas as pd
import numpy as np
from sklearn import model_selection



class CrossValidation:

    def __init__(self, df, target_cols, shuffle, multilabel_delimiter = ",", problem_type = "binary_classification", num_folds = 5,  random_state = 1):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.multilabel_delimiter = multilabel_delimiter
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state   
        if self.shuffle == True:
            self.dataframe = self.dataframe.sample(frac = 1).reset_index(drop=True)

        self.dataframe["kfold"] = -1

        return

    def split(self):
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")

            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values <= 1:
                raise Exception("Only one unique value found")

            elif unique_values > 1:
                
                kf = model_selection.StratifiedKFold(n_splits=5, shuffle=self.shuffle, random_state=self.random_state)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X = self.dataframe, y = self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, "kfold"] = fold

        elif self.problem_type in ["single_col_regression", "multi_col_regression"]:
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets > 1 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")

            kf = model_selection.KFold(n_splits = self.num_folds)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X = self.dataframe)):
                self.dataframe.loc[val_idx, "kfold"] = fold

        elif self.problem_type.startswith("holdout_"):
            #holdout_1, holdout_2, holdout_3
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = (len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1


        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")

            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=5, shuffle=self.shuffle, random_state=self.random_state)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X = self.dataframe, y = targets)):
                self.dataframe.loc[val_idx, "kfold"] = fold



        else:
            raise Exception("Problem  type not understood!")

        return self.dataframe

if __name__ == "__main__":
    df = pd.read_csv("../input/train_multilabel.csv")
    cv = CrossValidation(df, target_cols = ["attribute_ids"], problem_type = "multilabel_classification", shuffle = True, multilabel_delimiter = " ")
    df_split = cv.split()

    print(df_split.head())
    print(df_split.kfold.value_counts())

"""
MUlti label classification

id, target
1, 1,0,2
2, 0,1
3, 4,5
4, 7
"""