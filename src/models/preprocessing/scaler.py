import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer


num_cols = ['AGLOMERADO', 'CH06', 'IV2', 'II2', 'IX_TOT', 
            'IX_MEN10', 'CH06_jefx', 'ratio_ocupados',
            'PONDERA']

def make_scaler(cols: list[str] | None = None):
    """Generates a standard scaler for numeric columns.

    Args:
        cols (list, optional): List of numeric columns to scale. Defaults to None.

    Returns:
        sklearn.compose.ColumnTransformer: A StandardScaler transformer.
    """    
    if cols == None:
        cols = num_cols
    stdscaler = make_column_transformer(
        (StandardScaler().set_output(transform='pandas'), cols),
        remainder='passthrough',
        verbose_feature_names_out=False   
    )
    return stdscaler.set_output(transform='pandas')


if __name__ == '__main__':
    data_path = '~/Code/data/desercionAR/train.csv'
    data = pd.read_csv(data_path)
    scaler = make_scaler()
    scaler.fit_transform(data)
    scaled = pd.DataFrame(scaler.fit_transform(data), 
                           columns=scaler.get_feature_names_out())
    save_path = '~/Code/data/desercionAR/train_scaled.csv'
    scaled.to_csv(save_path, index=False)
