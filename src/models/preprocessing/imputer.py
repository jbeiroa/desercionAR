import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.compose import make_column_transformer

cols9 = ['CH07', 'CH08', 'CH11', 'V1', 
        'V2', 'V3', 'V5', 'V6', 'V7', 'V8', 'V11', 'V12', 
        'V13', 'V14', 'PP07I_jefx']

cols99 = ['IV2', 'II1']

col12 = ['DECCFR']

def make_imputer(c9: list[str] | None = None, 
                 c99: list[str] | None = None, 
                 c12: list[str] | None = None):
    """Makes an imputer for columns with specific missing value codes.

    Args:
        c9 (list, optional): List of columns with missing values coded as 9. Defaults to None.
        c99 (list, optional): List of columns with missing values coded as 99. Defaults to None.
        c12 (list, optional): List of columns with missing values coded as 12. Defaults to None.

    Returns:
        sklearn.compose.ColumnTransformer: A KNN imputer transformer.
    """
    if c9 is None:
        c9 = cols9
    if c99 is None:
        c99 = cols99
    if c12 is None:
        c12 = col12    
    knnimputer = make_column_transformer(
        (KNNImputer(n_neighbors=1, missing_values=9.).set_output(transform='pandas'), c9),
        (KNNImputer(n_neighbors=1, missing_values=99.).set_output(transform='pandas'), c99),
        (KNNImputer(n_neighbors=1, missing_values=12.).set_output(transform='pandas'), c12),
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    return knnimputer.set_output(transform='pandas')

if __name__ == "__main__":
    data_path = '~/Code/data/desercionAR/train.csv'
    data = pd.read_csv(data_path)
    imputer = make_imputer(cols9, cols99, col12)
    imputer.fit_transform(data)
    imputed = pd.DataFrame(imputer.fit_transform(data), 
                           columns=imputer.get_feature_names_out()).round(0)
    save_path = '~/Code/data/desercionAR/train_imputed.csv'
    imputed.to_csv(save_path, index=False)
