from ikrlib import wav16khz2mfcc
from sklearn.mixture import GaussianMixture
import sys
import train_gmm as train
import getopt
import numpy as np


argv = sys.argv[1:]
opts, temp = getopt.getopt(argv, "", ["tdir=", "ntdir=", "testdir="])

for arg, value in opts:
    if arg == "--tdir":
        target_dir = value
    elif arg == "--ntdir":
        nontarget_dir = value
    else:
        test_dir = value

target_gmm = train.target_gmm(target_dir)
gmms = train.non_target_gmms(nontarget_dir, 10)

#score_list = []
count = 0
result = open("result.txt", "w")
test_dict = wav16khz2mfcc(test_dir)
for key, feature in sorted(test_dict.items()):
    result_line = ""
    name = key.split(".")[0]
    name = name.split("/")[1]
    result_line += name + " "
    score = sum(target_gmm.score_samples(feature))
    scores2 = []
    for gmm in gmms:
        scores2.append(sum(gmm.score_samples(feature)))
    maximum = max(scores2)
    result_line += str(int((score+np.log(0.5)) - (maximum+np.log(0.5))))+ " "
    if maximum > score:
        result_line += "0"
    else:
        result_line += "1"
        count += 1
    result.write(result_line+"\n")

result.close()
print(count)


"""print("Found: "+ str(score_list.count(1)))
print("No:" + str(score_list.count(0)))"""
