from snorkel.labeling import LFAnalysis
from labelling_functions import *
from applying_lfs import *



'''
The analyze_labelling_performance function returns the performance of the written labelling
functions.

Params:
    lfs (list): This is a list of the written labelling functions
    L_train (obj): This is 
'''
def analyze_labelling_performance(lfs, L_train):
    return LFAnalysis(L=L_train, lfs=lfs).lf_summary()


#polarities demonstrate which labels a labelling function has omitted
#when given some data
def generate_polarities(lfs, L_train):
    return LFAnalysis(L=L_train, lfs=lfs).lf_polarities()


#coverages demonstrate how often the labelling functions have an opinion
#about the value of a given data point within the data
def generate_coverage(lfs, L_train):
    return LFAnalysis(L=L_train, lfs=lfs).lf_coverages()


#overlaps return a measure of how often different labelling functions
#vote the same way on a given data point within the dataset
def generate_overlaps(lfs, L_train):
    return LFAnalysis(L=L_train, lfs=lfs).lf_overlaps()


#conflicts return a measure of how often different labelling functions disagree
#on the label of the same data point 
def generate_conflicts(lfs, L_train):
    return LFAnalysis(L=L_train, lfs=lfs).lf_conflicts()


