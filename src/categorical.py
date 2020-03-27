from sklearn import preprocessing
from sklearn import linear_model
import pandas as pd
"""
Excersizes
# Implement labelcount encoding
# Implement count encoding
# Implement target encoding
# Optimize binarizer encoding
"""
class CategoricalFeatures:

    def __init__(self, df, categorical_features, encoding_type = "label_encoding", handle_na = False):
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

        if handle_na == True:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna("-1")
        self.output_df = self.df.copy(deep = True)
        
    def label_encoding(self):
        for c in self.cat_feats:
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
    def binarizer_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values) #numpy array output
            self.output_df = self.output_df.drop(c, axis = 1) # remove old columns
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    def ohe_encoding(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats])
        return ohe.transform(self.df[self.cat_feats].values)

    def fit_transform(self):
        if self.enc_type == "label":
            return self.label_encoding()
        elif self.enc_type == "binarizer":
            return self.binarizer_encoding()
        elif self.enc_type == "ohe":
            return self.ohe_encoding()
        else:
            raise Exception("Encoding not understood")

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
    cat_feats = CategoricalFeatures(full_data, categorical_features=cols, 
                                    encoding_type="ohe", handle_na=True)
    
    full_data_transformed = cat_feats.fit_transform()

    x_train = full_data_transformed[:train_len, :]
    x_test = full_data_transformed[train_len:, :]

    #train_df = full_data_transformed[full_data_transformed["id"].isin(train_idx)].reset_index(drop = True)
    #test_df = full_data_transformed[full_data_transformed["id"].isin(test_idx)].reset_index(drop = True)

    print(x_train.shape)
    print(x_test.shape)

    clf = linear_model.LogisticRegression()
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
