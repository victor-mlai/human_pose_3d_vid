import cv2
import numpy as np
import pickle as pkl

annotations = np.load("dataset/U3DPW/annotation.npy", allow_pickle=True)
for annotaion in annotations:
    pass

with open("dataset/3DPW/sequenceFiles/train/courtyard_golf_00.pkl", "rb") as f:
    seq = pkl.load(f)
    pass

image_dog = cv2.imread("./dataset/3DPW/imageFiles/courtyard_box_00/image_00328.jpg")
