self.X_train, self.y_train = self._load_train_data(self.config['train_data_path'])

def _load_train_data(self, train_data_path) -> tuple[pd.DataFrame, np.ndarray]:
    id_cols = ['CODUSU', 'NRO_HOGAR', 'COMPONENTE', 'ANO4', 'TRIMESTRE']
    data = pd.read_csv(train_data_path)
    train_data = data.loc[:, ~data.columns.isin(id_cols)]
    X_train = train_data.loc[:, train_data.columns != 'DESERTO']
    y_train = train_data.loc[:, train_data.columns == 'DESERTO'].values.ravel()
    return X_train, y_train