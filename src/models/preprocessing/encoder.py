import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

columnas_cat_nominales = [
    'REGION', 'CH03', 'CH07', 'CH15', 'CH09', 'CH16', 'ESTADO', 
    'ESTADO_jefx', 'ESTADO_conyuge', 'PP02E'
]

def make_encoder(cols: list[str] | None = None):
    """Makes a one-hot encoder column transformer for categorical features.

    Args:
        cols (list, optional): List of categorical columns to encode. Defaults to None.

    Returns:
        sklearn.compose.ColumnTransformer: A OneHotEncoder transformer.
    """    
    if cols is None:
        cols = columnas_cat_nominales
    ohencoder = make_column_transformer(
        (OneHotEncoder(drop='first', 
                       sparse_output=False,
                       handle_unknown='infrequent_if_exist'), cols),
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    return ohencoder


if __name__ == '__main__':
    data_path = '~/Code/data/desercionAR/train.csv'
    data = pd.read_csv(data_path)
    encoder = make_encoder()
    encoder.fit_transform(data)
    encoded = pd.DataFrame(encoder.fit_transform(data), 
                           columns=encoder.get_feature_names_out())
    save_path = '~/Code/data/desercionAR/train_encoded.csv'
    encoded.to_csv(save_path, index=False)
