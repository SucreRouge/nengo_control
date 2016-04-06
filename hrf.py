import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from scipy.stats import gamma

def hrf(t):
    peak = gamma.pdf(t, 6) #gamma pdf for peak
    undershoot = gamma.pdf(t, 12) #gamma pdf for undershoot
    vals = peak - 0.35 * undershoot
    return vals/np.max(vals) * 0.6
