from models.nn.SkDnn import SkDnn


def create_dnn(features, fingerprints_columns, descriptors_columns, binary_columns):
    if features == "fingerprints":
        return SkDnn(use_col_indices=fingerprints_columns, binary_col_indices=binary_columns, transform_output=True)
    elif features == "descriptors":
        return SkDnn(use_col_indices=descriptors_columns, binary_col_indices=binary_columns, transform_output=True)
    else:
        return SkDnn(use_col_indices='all', binary_col_indices=binary_columns, transform_output=True)