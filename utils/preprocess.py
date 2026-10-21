import pandas as pd


def prepare_features(df, feature_columns):
    X = df.drop(columns=["Attrition", "EmployeeID"], errors="ignore")
    cat_cols = X.select_dtypes(include="object").columns
    X = pd.get_dummies(X, columns=cat_cols, dtype=int)
    return X.reindex(columns=feature_columns, fill_value=0)
