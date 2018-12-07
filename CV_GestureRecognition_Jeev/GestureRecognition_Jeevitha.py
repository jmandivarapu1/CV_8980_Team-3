########################################################################
# Project Name - Hand Gesture Recognition
# Scripted By - Jeevitha Meyyappan
# Presentation Date - 28.11.2018
# Subject - Computer Vision
########################################################################

import cv2
import numpy as np
import math
 
#Open the Laptop Camera 
camera = cv2.VideoCapture(0)
while(camera.isOpened()):
     
    ret, img = camera.read()

    # Define the rectangle
    cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)

    # Define the region
    crop_img = img[100:300, 100:300]

     # Provide value for the blur 
    blurValue = (35, 35)

    # Extract the grey image
    greyImage = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Blur the image with GaussianBlur
    blurImage = cv2.GaussianBlur(greyImage, blurValue, 0)

    # Apply Threshold on the Image
    _, thresh1 = cv2.threshold(blurImage, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Display Threshold 
    cv2.imshow('Thresholded', thresh1)

    #Extract the Version
    (version, _, _) = cv2.__version__.split('.')

    # Find contours based on the OpenCV version
    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    # Find contour of max area(hand)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # Find the bounding rectangle 
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # Make convex hull around hand
    hull = cv2.convexHull(cnt)

    # Define area of hull and area of hand
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)
      
    # Find the percentage of area not covered by hand in convex hull
    arearatio=((areahull-areacnt)/areacnt)*100

    #drawing contours
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    #Create a ConvexHull
    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
 
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        Finger_start = tuple(cnt[s][0])
        Finger_end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        pt= (100,180)

        # Apply the Math functions
        a = math.sqrt((Finger_end[0] - Finger_start[0])**2 + (Finger_end[1] - Finger_start[1])**2)
        b = math.sqrt((far[0] - Finger_start[0])**2 + (far[1] - Finger_start[1])**2)
        c = math.sqrt((Finger_end[0] - far[0])**2 + (Finger_end[1] - far[1])**2)
        s = (a+b+c)/2
        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

        # Distance between point and convex hull
        d=(2*ar)/a

        # Apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # Ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
        if angle <= 90 and d>30:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)
         
        cv2.line(crop_img,Finger_start, Finger_end, [0,255,0], 2)
        
    if count_defects == 0:
        
        if areacnt<2000:
            cv2.putText(img,'Put hand in the box',(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            if arearatio<12:
                cv2.putText(img,"Value is 0", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
            elif arearatio<17.5:
                cv2.putText(img,"Move you hand and Try again!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
            elif arearatio>17.5 and arearatio<23:
                cv2.putText(img,"Value is 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                cv2.putText(img,"Reposition", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
       
    elif count_defects == 1:
        cv2.putText(img,"Value is 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
    elif count_defects == 2:
        cv2.putText(img,"Value is 3", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
    elif count_defects == 3:
        cv2.putText(img,"Value is 4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
    elif count_defects == 4:
        cv2.putText(img,"Value is 5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
    else :
        cv2.putText(img,'Reposition your hand',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)

    #Show the Gesture
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))

    #Show the Contours
    cv2.imshow('Contours', all_img)

    k = cv2.waitKey(10)
    if k == 27:
        break
