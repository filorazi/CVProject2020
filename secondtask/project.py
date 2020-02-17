import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from copy import copy
from operator import itemgetter
from scipy.spatial import distance


dict = {}
samp = {}
#insert images in a Dict
#this is done once because of the small number of images available
for file in os.listdir("."):
    if file.endswith(".png") and file.startswith("C"):
        if file[3:9] not in dict.keys():
            dict[file[3:9]] = {}
            dict[file[3:9]][file[0:2]] = cv2.imread( file, cv2.IMREAD_GRAYSCALE)
        else:
            dict[file[3:9]][file[0:2]] = cv2.imread( file, cv2.IMREAD_COLOR)
            dict[file[3:9]][file[0:2]] = cv2.cvtColor(dict[file[3:9]][file[0:2]],cv2.COLOR_BGR2HSV)

#Putting the sample in a dict and compute covariance and mean of the samples
for file in os.listdir("."):
    if file.endswith(".png") and file.startswith("S"):
        samp[file[3:9]] = {}
        samp[file[3:9]][file[0:2]] = cv2.imread( file, cv2.IMREAD_COLOR)
        samp[file[3:9]][file[0:2]] = cv2.cvtColor(samp[file[3:9]][file[0:2]],cv2.COLOR_BGR2HSV)

final_covar = np.zeros((3,3))
final_mean = np.zeros(3)
samp["000004"]["S1"] = samp["000004"]["S1"].reshape(6136,3)

for key in samp.keys():
    final_covar += np.cov(samp[key]["S1"],rowvar=False, bias=True)
    final_mean[0] += np.mean(list( map(itemgetter(0), samp[key]["S1"])))
    final_mean[1] += np.mean(list( map(itemgetter(1), samp[key]["S1"])))
    final_mean[2] += np.mean(list( map(itemgetter(2), samp[key]["S1"])))

final_mean = final_mean/len(samp.keys())
final_covar = cv2.invert(final_covar, cv2.DECOMP_SVD)



for key in dict.keys():
    #finding a binarized version, that is used to delete the background
    frt, dict[key]["mask"] = cv2.threshold(dict[key]["C0"],35,255,cv2.THRESH_BINARY)
    kernel = np.ones((7,7),np.uint8)
    dict[key]["mask"] = cv2.morphologyEx(dict[key]["mask"], cv2.MORPH_ERODE, kernel)
    dict[key]["mask"] = cv2.morphologyEx(dict[key]["mask"], cv2.MORPH_ERODE, kernel)

    kernel = np.ones((5,5),np.uint8)
    dict[key]["mask"] =  cv2.morphologyEx(cv2.morphologyEx(dict[key]["mask"], cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    contours = cv2.findContours(dict[key]["mask"],cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    dict[key]["mask"] ^= dict[key]["mask"]
    cv2.drawContours(dict[key]["mask"], contours, 0, (255,255,255), cv2.FILLED,8)
    dict[key]["Gdiff"] = cv2.subtract(dict[key]["mask"],255-dict[key]["C0"])
    dict[key]["Cdiff"] = cv2.subtract(cv2.cvtColor(dict[key]["mask"], cv2.COLOR_GRAY2RGB),255-dict[key]["C1"])

    #find each pixel with a small enough distance and clean the blob with some morphology operations
    dict[key]["Stain"] = dict[key]["mask"]^dict[key]["mask"]
    d = []
    for i in range(len(dict[key]["Cdiff"])):
        row = dict[key]["Cdiff"][i]
        for j in range(len(row)):
            if dict[key]["Cdiff"][i][j].any():
                #dist = distance.mahalanobis(row[j], final_mean, covariance)
                dist = distance.euclidean(row[j],final_mean)
                d.append(dist)
                if  dist < 20:
                    dict[key]["Stain"][i][j] = int(255)
                else:
                    dict[key]["Stain"][i][j] = int(0)

    dict[key]["Stain"] =  cv2.morphologyEx(cv2.morphologyEx(dict[key]["Stain"], cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    dict[key]["Stain"] = cv2.morphologyEx(dict[key]["Stain"], cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,11)))
    frt , dict[key]["Stain"] = cv2.threshold(dict[key]["Stain"],35,255,cv2.THRESH_BINARY)

    #highlight the stain in the apple
    contours = cv2.findContours(dict[key]["Stain"],cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    dict[key]["Cdiff"] = cv2.cvtColor(dict[key]["Cdiff"], cv2.COLOR_HSV2RGB)
    for c in contours:
        M = cv2.moments(c)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(dict[key]["Cdiff"],(x-10,y-10),(x+20+w,y+h+20),(0,255,0),2)
    plt.imshow(dict[key]["Cdiff"])
    plt.show()
