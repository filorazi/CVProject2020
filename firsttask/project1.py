import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


dict = {}
#insert images in a printDictFigures
#this is done once because of the small number of images available
for file in os.listdir("."):
    if file.endswith(".png"):
        if file[3:9] not in dict.keys():
            dict[file[3:9]] = {}
            dict[file[3:9]][file[0:2]] = cv2.imread( file, cv2.IMREAD_GRAYSCALE)
        else:
            dict[file[3:9]][file[0:2]] = cv2.imread( file, cv2.IMREAD_COLOR)
            dict[file[3:9]][file[0:2]] = cv2.cvtColor(dict[file[3:9]][file[0:2]],cv2.COLOR_BGR2RGB)

for key in dict.keys():
    #adding a binarized version, that is used to delete the background trough the countour
    frt, dict[key]["mask"] = cv2.threshold(dict[key]["C0"],35,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    dict[key]["mask"] =  cv2.morphologyEx(cv2.morphologyEx(dict[key]["mask"], cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    contours = cv2.findContours(dict[key]["mask"],cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    dict[key]["mask"] ^= dict[key]["mask"]
    cv2.drawContours(dict[key]["mask"], contours, 0, (255,255,255), cv2.FILLED,8)

    #creating Grayscale and colorimage without the background, blurring the grayscale
    dict[key]["Gdiff"] = cv2.subtract(dict[key]["mask"],255-dict[key]["C0"])
    dict[key]["Gdiff"] = cv2.blur(dict[key]["Gdiff"],(3,3))
    dict[key]["Cdiff"] = cv2.subtract(cv2.cvtColor(dict[key]["mask"], cv2.COLOR_GRAY2RGB),255-dict[key]["C1"])

    #finding the edge image and the binary image containing the anomalies
    dict[key]["edge"] = cv2.Canny(dict[key]["Gdiff"],100,100)
    dict[key]["binFr"] = dict[key]["edge"] - (255 - dict[key]["mask"])
    cv2.drawContours(dict[key]["binFr"], contours, 0, 0, 8)

    #the binary immage is modified to clean the shape of the anomalies
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
    dict[key]["binFr"] = cv2.morphologyEx(dict[key]["binFr"], cv2.MORPH_CLOSE, kernel)
    dict[key]["binFr"] = cv2.morphologyEx(dict[key]["binFr"], cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
    frt , dict[key]["binFr"] = cv2.threshold(dict[key]["binFr"],35,255,cv2.THRESH_BINARY)

    #the anomalies are highlighted in the colorimage
    contours = cv2.findContours(dict[key]["binFr"],cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    for c in contours:
        M = cv2.moments(c)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(dict[key]["Cdiff"],(x-10,y-10),(x+20+w,y+h+20),(0,255,0),2)


    plt.imshow(dict[key]["Cdiff"],cmap="gray")
    plt.show()
