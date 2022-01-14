## importing the library

import cv2 
import numpy as np 
import timeit
import math


## calculating the distance between two points 
def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
### Estimatig the nearest point 
def closest(pt, others):
    return min(others, key = lambda i: distance(pt, i))

#add = r"D:\Ananya\3RDi_LABs\SubElement\v3.mp4"

brisk = cv2.ORB_create()        
brisk = cv2.BRISK_create()

totalNumberofSubElement = 1

s1 = r"snapshot.jpg" ### Address of the image to draw the bounding box around the painting 
img = cv2.imread(s1)
h, w, c = img.shape
imgsave =  img 
point = []
im = np.zeros([h,w],dtype=np.uint8)
im.fill(255)
roi = cv2.selectROI(img)
subimg = img[roi[1]:roi[3]+roi[1],roi[0]:roi[2]+roi[0]]

points = (int(roi[0]+(roi[2]/2)) , int(roi[1]+(roi[3]/2)))
imgsave = cv2.circle(imgsave, points , 8 , (125, 0, 0), -1)
im = cv2.circle(im, points , 30 , (125), -1)
im = cv2.resize(im,(int(img.shape[1]/4),int(img.shape[0]/4)))
cv2.imshow('Frame', im) 
cv2.waitKey(0)
print(cv2.imwrite(r"hotspot" + ".jpg", imgsave)) ## saving the hotspot

x = []
y = []
for i in range(len(im)):
    for j in range(len(im[i])):
        
        if im[i][j] == 125:
            x.append(j)
            y.append(i)
        
point.append([int((min(x)+max(x))/2) , int((min(y)+max(y))/2)])
print(point, points)
img = cv2.imread(s1)
h, w, c = img.shape
im = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
im = cv2.circle(im, (int((min(x)+max(x))/2) , int((min(y)+max(y))/2)) , 6 , (125, 0, 0), -1)
    
cv2.imshow('Frame', im) 
cv2.waitKey(0)

from numpy.linalg import norm 
def pointLineDis(a , linepoint):
    b,c = linepoint
    

    ba = a - b 
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    dis_ = distance(a,b)
    return dis_ * cosine_angle



img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
start = timeit.default_timer()
kp1, des1 = brisk.detectAndCompute(img, None)
stop = timeit.default_timer()
h3,w3,c2 = img.shape
print(len(kp1))
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)  
flann = cv2.FlannBasedMatcher(index_params,search_params)
font = cv2.FONT_HERSHEY_SIMPLEX 

add = r"video.avi" ## video address

cap = cv2.VideoCapture(add) 
if (cap.isOpened()== False): 
    print("Error opening video file") 

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
   

#tracking using kernelized correlational filters 

tracker = []
bbox = []
for i in range(totalNumberofSubElement):
    tracker.append(cv2.TrackerKCF_create())




found  = 0





result = cv2.VideoWriter(r"SavedVideo.avi",  cv2.VideoWriter_fourcc(*'MJPG'), 10, size)   ##

while(cap.isOpened()): 
    ret, frame = cap.read() 

    if ret == True:
        #image = frame
        
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = frame
        yu = frame
        #result.write(yu)
        
        if found == 0 :
            h_, w_,c = frame.shape
            #frame = cv2.resize(frame,(int(frame.shape[1]/4),int(frame.shape[0]/4))) ## downsizing only when the size of the image is too small 
            

            start = timeit.default_timer()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = brisk.detectAndCompute(frame, None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1,des, k=2)
            good = []
            
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.75*n.distance:  
                    good.append(m)
            print(len(good))
            if len(good) >  10 :
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)







                s_ = [(int(b[0][0]),int(b[0][1])) for b in src_pts]
                d_ = [(int(b[0][0]),int(b[0][1])) for b in dst_pts]
                origin = [0,0]
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                h,w,c = img.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                point1 = np.int32(dst)
                w1 = point1[2][0][0] - point1[1][0][0]
                w2 = point1[3][0][0] - point1[0][0][0]
                h1 = point1[1][0][1] - point1[0][0][1]
                h2 = point1[2][0][1] - point1[3][0][1]



                for i in range(totalNumberofSubElement):
                    p = closest(point[i], s_)
        
                    #print(p , point, (start-stop))
                    indx = s_.index(p)
                    point2 = d_[indx]
                    # p 
                    d1 = pointLineDis(p,[point1[2][0], point1[3][0]])
                    d2 = pointLineDis(p,[point1[1][0], point1[0][0]])
                    d3 = pointLineDis(p,[point1[2][0], point1[1][0]])
                    d4 = pointLineDis(p,[point1[3][0], point1[0][0]])

                    h = ( d1*h2 + d2*h1 )/(d1 + d2)
                    w = ( d3*w1 + d4*w2 )/(d3 + d4)

                    
                    x2 = int((point[i][0] - p[0])*(h/h3) + point2[0])
                    y2 = int((point[i][1] - p[1])*(w/w3) + point2[1])
                    #real coordinates
                    


                    
                    x3 = int(x2*(h_/frame.shape[0]))
                    y3 = int(y2*(w_/frame.shape[1]))


                    
                    #x3 = int(x2*(h_/frame.shape[0]))
                    #y3 = int(y2*(w_/frame.shape[1]))
                    bb = (x3 - 50 , y3 - 50 , 100, 100)
                    ok = tracker[i].init(image, bb)
                    bbox.append(bb)

                    image = cv2.circle(image, (x3,y3), 6 , (255), -1)
                found = 1
        else:
            #track 
            #print("hfbvjdc")
            for i in range(totalNumberofSubElement):
                #print("tracking")
                timer = cv2.getTickCount()
                ok, bbox[i] = tracker[i].update(image)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                if ok:
                    #p1 = (int(bbox[i][0]), int(bbox[i][1]))
                    print("tracking")
                    center = (int(bbox[i][0] + (bbox[i][2]/2)), int(bbox[i][1] + (bbox[i][3]/2)))
                    
                    image = cv2.circle(image, center, 6,  (255), -1)
                else:
                    found = 0
                    print("NOt found in next")
                    

        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
        result.write(image)
        
        cv2.imshow('Frame', image) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
          
    else: 
        break

#video.release()
result.release()
 
cap.release()  
cv2.destroyAllWindows() 
