import cv2
import time

img_10 = cv2.imread("inf.png")
img_20 = cv2.imread("20.png")
img_30 = cv2.imread("30.png")
img_40 = cv2.imread("40.png")

while(True):
    cv2.imshow("10", img_10)
    #cv2.imshow("20", img_20)
    #cv2.imshow("30", img_30)
    #cv2.imshow("40", img_40)
    
    cv2.waitKey(5)

#10 291 3.436
#20 194 5.154
#30 174 5.747
#40 161 6.211