from sklearn import preprocessing
from sklearn import linear_model
from scipy import sparse
import pandas as pd
import numpy as np
"""
Excersizes
# Implement labelcount encoding
# Implement count encoding
# Implement target encoding
# Optimize binarizer encoding
"""
class CategoricalFeatures:

    def __init__(self, df, categorical_features, 
                ohe_features = None, label_encoding_features = None,
                binarizer_features = None, count_encoding_features = None, labelcount_encoding_features = None, 
                encoding_type = "label_encoding", handle_na = False):
        """
        df: pandas dataframe
        categorical column list: list of column names
        encoding type : label, ohe, binarizer
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None
        self.count_encoders = dict()
        self.labelcount_encoders = dict()

        ## Initializtng feature columns for different encoding
        self.ohe_feats = ohe_features
        self.label_encoding_feats = label_encoding_features
        self.binarizer_feats = binarizer_features
        self.count_encoding_feats = count_encoding_features
        self.labelcount_encoding_feats = labelcount_encoding_features

        if handle_na == True:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna("-1")
        self.output_df = self.df.copy(deep = True)
        
    def label_encoding(self, label_encoding_feats):
        for c in label_encoding_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    """
    In label binarizer, instead of createing new columns in for loop (j)
    keep track of which columns you saw using col
    and then just stack the vales togather in new array
    instead of returning a dataframe return an numpy array
    """  
    def binarizer_encoding(self, binarizer_feats):
        for c in binarizer_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values) #numpy array output
            self.output_df = self.output_df.drop(c, axis = 1) # remove old columns
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    def ohe_encoding(self, ohe_feats):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[ohe_feats])
        return ohe.transform(self.df[ohe_feats].values)

    def fit_transform(self):
        encoded_cols_dict = dict()
        for key, value in self.enc_type.items():
            if key == "label":
                encoded_cols_dict[key] = self.label_encoding(value)
            elif key == "binarizer":
                encoded_cols_dict[key] = self.binarizer_encoding(value)
            elif key == "ohe":
                encoded_cols_dict[key] = self.ohe_encoding(value)
            else:
                raise Exception("Encoding not understood while fitting")
        return encoded_cols_dict

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-1")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binarizer":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                
                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        elif self.enc_type == "ohe":
            self.ohe = self.ohe(dataframe[self.cat_feats].values)
            return self.ohe
        else:
            raise Exception("Encoding type not understood while transforming")
        

if __name__ == "__main__":
    df = pd.read_csv("../input/train_cat.csv")
    df_test = pd.read_csv("../input/test_cat.csv")
    sample = pd.read_csv("../input/sample_submission_cat.csv")
    
    train_idx = df["id"].values
    test_idx = df_test["id"].values

    train_len = len(df)
    test_len = len(df_test)


    df_test["target"] = -1
    full_data = pd.concat([df, df_test], axis = 0)

    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    
    ohe_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 
                                'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 
                                'nom_8', 'nom_9']
    label_encoder_features  = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
    encoding_type = {"label": label_encoder_features, "ohe": ohe_features}

    cat_feats = CategoricalFeatures(full_data, categorical_features=cols, 
                                    encoding_type = encoding_type, ohe_features = ohe_features, 
                                    label_encoding_features = label_encoder_features, handle_na=True)

    full_data_transformed_dict = cat_feats.fit_transform()
    print(full_data_transformed_dict["label"].describe())
    print(full_data_transformed_dict["label"].head())
    print(full_data_transformed_dict["label"].count)

    tmp_df = full_data_transformed_dict["label"].copy()
    tmp_df.index = full_data_transformed_dict["label"]["id"]
    tmp_df = tmp_df[label_encoder_features]
    print(tmp_df.shape)
    print(tmp_df.head())
    x_train_label = tmp_df.loc[ : train_len-1, :].to_numpy(copy = True)
    x_test_label= tmp_df.loc[train_len : , :].to_numpy(copy = True)
    print("==============================")
    print("")
    print(x_test_label.shape)
    print(x_train_label.shape)
    x_train_ohe = full_data_transformed_dict["ohe"][:train_len, :]
    x_test_ohe = full_data_transformed_dict["ohe"][train_len:, :]
    print("==============================")
    print("")
    print(x_train_ohe.shape)
    print(x_test_ohe.shape)
    x_train = sparse.hstack((x_train_label, x_train_ohe))
    x_test = sparse.hstack((x_test_label, x_test_ohe))

    #train_df = full_data_transformed[full_data_transformed["id"].isin(train_idx)].reset_index(drop = True)
    #test_df = full_data_transformed[full_data_transformed["id"].isin(test_idx)].reset_index(drop = True)

    print(x_train.shape)
    print(x_test.shape)
    
    clf = linear_model.LogisticRegression(max_iter = 110, n_jobs = 4)
    clf.fit(x_train, df.target.values)
    preds = clf.predict_proba(x_test)[:, 1]
    sample.loc[:, "target"] = preds
    sample.to_csv("../input/submission_cat1.csv", index = False)
"""
    train_transformed = cat_feats.fit_transform()
    test_transformed = cat_feats.transform(df_test)
    print(train_transformed.head())
    print(test_transformed.head())
"""
