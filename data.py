import os.path
from os import listdir
from cv2 import cv2


i = 0
raw_folder = "raw_number/"

for file in listdir(raw_folder):
    count = 0
    if file != ".DS_store":
        print("file: ", file)
        while True:
            img = cv2.imread(r"{}{}".format(raw_folder, file))
            count += 1
            if count <= 500:
                print("Image Captured: ", count)
                if not os.path.exists("Data/" + str(i)):
                    os.mkdir("Data/" + str(i))

                cv2.imwrite("{}{}{}{}{}{}{}".format("Data/", str(i), "/", str(i), "_", count, ".png"), img)
            else:
                break
    i += 1


