import pandas as pd
from scripts.ClassyFireQuery import access_data
from scripts.descriptors_and_fingerprints import des_and_fgp
from scripts.descriptors_and_fingerprints import get_smrt

final_data_nt = access_data(imputation=True)
final_data_nt.to_csv("../resources/RepoRT_classified_CCinformation.tsv", sep='\t', index=False)
des_and_fgp()
get_smrt()
