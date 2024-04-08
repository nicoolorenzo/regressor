# CMM-RT
This code implements methods for the accurate prediction of Retention Times 
(RTs) for a given Chromatographic Method (CM) using machine learning, as 
described in:

> García, C.A., Gil-de-la-Fuente, A., Barbas, C. et al. Probabilistic metabolite annotation using retention time prediction and meta-learned projections. J Cheminform 14, 33 (2022). https://doi.org/10.1186/s13321-022-00613-8. 


Used to create the yml file:
conda env export > environment.yml

You can create your environment as follows:
conda env create -f environment.yml --name cmmrt_env

How to activate your environment:
conda activate cmmrt_env
