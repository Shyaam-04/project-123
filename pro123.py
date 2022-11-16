import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import cv2
from PIL import Image
import PIL.ImageOps

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
n_classes = len(classes)

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=7500,test_size = 2500,random_state=9)
X_train = X_train/255
X_test = X_test/255

clf = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        u_left = (int(width/2-56),int(height/2-56))
        b_right = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,u_left,b_right,(0,255,0),2)
        roi = gray[u_left[1]:b_right[1],u_left[0]:b_right[0]]
        im_pil = Image.fromarray(roi)
        img_bw = im_pil.convert("L")
        img_bw_resized = img_bw.resize((28,28),Image.ANTIALIAS)
        img_bw_resized_inverted = PIL.ImageOps.invert(img_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(img_bw_resized_inverted,pixel_filter)
        img_bw_resized_inverted_scaled = np.clip(img_bw_resized_inverted-min_pixel,0,255)
        max_pixel = np.max(img_bw_resized_inverted)
        img_bw_resized_inverted_scaled = np.array(img_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print(test_pred)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1)==27:
            break

    except:
        pass

cap.release()
cv2.destroyAllWindows()
