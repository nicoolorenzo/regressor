





def preprocess_X(fingerprints_columns, descriptors_columns, train_split_X, train_split_y):
    p = 0
    cor_th = 0.9
    k = 'all'

    def fit(self, X, y=None):
        self._init_hidden_models()
        self._desc_preprocessor.fit(X, y)
        self._fgp_preprocessor.fit(X, y)
        return self

    def transform(self, X, y=None):
        X_desc_proc = self._desc_preprocessor.transform(X)
        X_fgp_proc = self._fgp_preprocessor.transform(X)
        new_X = np.concatenate([X_desc_proc, X_fgp_proc], axis=1)
        # Annotate which columns are related to descriptors an fingerprints after transformation. Also, annotate which
        # columns can be considered binary
        self.transformed_desc_cols = np.concatenate([
            np.arange(X_desc_proc.shape[1]),
            np.arange(new_X.shape[1]-3, new_X.shape[1])
        ])
        self.transformed_fgp_cols = np.arange(X_desc_proc.shape[1], new_X.shape[1], dtype='int')
        self.transformed_binary_cols = binary_features_cols(new_X)
        return new_X

    preprocessed_train_split_X = fit(train_split_X, train_split_y).transform(train_split_X)


    return preprocessed_train_split_X, transformed_binary_cols


def preprocess_y(fingerprints_columns, descriptors_columns, train_split_X, train_split_y):
    pass

