from snorkel.labeling.apply.pandas import PandasLFApplier
from labelling_functions import *
import pandas as pd



'''
The apply_lfs function applies the formulated labelling functions to the comments data stored in 
a given pandas dataframe

Params:
    pandas_dataframe (Obj): This is the pandas dataframe containing the comments.
    lfs (list): This is a list of the defined labelling functions

Returns:
    L_train (obj) : This represents a matrix with each column of the matrix representing a labeling
                    function and each row of the matrix representing a prediction of the corresponding 
                    labeling function. 
'''

def apply_lfs(pandas_dataframe, lfs):
    #apply the defined labelling functions to the comments in the dataframe
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=pandas_dataframe)
    return L_train