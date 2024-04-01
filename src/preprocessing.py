def preprocess_my_data(train_split_X, train_split_y):

    preprocessor = Preprocessor(
        desc_cols=desc_cols,
        fgp_cols=fgp_cols
    )
    preprocessed_train_split_X = preprocessor.fit_transform(train_split_X, train_split_y)

    return preprocessed_train_split_X, preprocessed_train_split_y