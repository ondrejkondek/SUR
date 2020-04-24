import getopt
from ikrlib import wav16khz2mfcc
from sklearn.mixture import GaussianMixture
import sys


def target_gmm(dir):
    target_dict = wav16khz2mfcc(dir)
    gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
    for key, value in target_dict.items():
        gmm.fit(value)
    return gmm

def non_target_gmms(dir, count):
    notarget_dict = wav16khz2mfcc(dir)
    gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
    gmms = []
    cnt = 1
    keys = sorted(notarget_dict)
    for key in keys:
        feature = notarget_dict[key]
        gmm.fit(feature)
        if cnt == count:
            gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
            gmms.append(gmm)
            cnt = 0
        cnt += 1
    return gmms


