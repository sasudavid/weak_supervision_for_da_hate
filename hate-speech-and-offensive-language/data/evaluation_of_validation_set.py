from snorkel.labeling.analysis import LFAnalysis


'''
The evaluate_validation_set function tests the efficiency of
the labelling functions at labelling the validation set.

Params:
    lfs: Labelling functions
    L_validation: Pandas dataframe containing the validation set data
    true_labels: Pandas dataframe containing the true labels of the data points in the 
                 validation set.

Return:
    Matrix object containing the polarity, overlaps, conflicts and coverage, 
    in addition to the efficiency, the correct and the incorrect statistics.
'''
def evaluate_validation_set(lfs, L_validation, true_labels):
    return LFAnalysis(L_validation, lfs).lf_summary(true_labels)


