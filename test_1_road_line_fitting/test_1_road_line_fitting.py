from math import *
import numpy as np
import matplotlib.pyplot as plt
import cv2


######################## PART 1 : IMAGE PRE-PROCESSING ########################

road_image = cv2.imread('data\\input\\001109.png')
image0 = road_image.copy()

cv2.imshow("road", image0) 
cv2.waitKey(0)                    
cv2.destroyAllWindows()  
  
def preprocessing(image):
    # Gray conversion
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gaussian blur to decrease noise to reduce false edges
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # Canny edge detectiom
    img_edges = cv2.Canny(img_blur, 100, 200)
    return img_edges

image1=preprocessing(image0)

cv2.imshow("Road edges", image1) 
cv2.waitKey(0)                    
cv2.destroyAllWindows()  


############# PART 2 : HOUGH TRANSFORM FOR TRAFFIC LINES DETECTION ############

# Hough acumulator H(θ, ρ)
def compute_H(image): 
    # Initialize accumulator H  to all zeros
    width, height = image.shape[:2]
    diag = int(sqrt(width**2 + height**2))
    H_acc = np.zeros((180, diag))
    
    # For each feature point (x,y) in the image
    #   For θ in (0,180)
    #   ρ = x*cos(θ) + y*sin(θ)
    #   H(θ, ρ) = H(θ, ρ) + 1
    edges = np.nonzero(image)  # (row, col) indexes to edges points in the image
    for (x,y) in np.transpose(edges) :
      for theta in range(len(H_acc)) :
          rho = int(x*cos(theta*pi/180) + y*sin(theta*pi/180))
          H_acc[theta, rho]=H_acc[theta, rho]+1
    
    return H_acc

# Find the value(s) of (θ, ρ) where H(θ, ρ) is a local maximum
H = compute_H(image1)

def find_indexes(H):
    threshold=(np.max(H))/2
    indexes=np.zeros(shape=(1,2))
    for i in range(len(H)):
        for j in range(1, len(H[0])-1):
            if H[i,j]>H[i,j-1] and H[i,j]>H[i,j+1] and H[i,j]>threshold:
                indexes=np.append(indexes,[[i,j]], axis=0)
    indexes=np.delete(indexes,0,axis=0)
    return(indexes)

H_indexes=find_indexes(H)


############################# PART 3 : VISUALIZATION ##########################

width, height = image1.shape[:2]
# The detected line in the image is given by ρ = x cos θ + y sin θ
for (theta, rho) in H_indexes:
    cos_thet=cos(theta*pi/180)
    sin_thet=sin(theta*pi/180)
    x0 = cos_thet*rho 
    y0 = sin_thet*rho
    x1 = int(x0 + 1000*(-sin_thet))
    y1 = int(y0 + 1000*(cos_thet))
    x2 = int(x0 - 1000*(-sin_thet))
    y2 = int(y0 - 1000*(cos_thet))

    cv2.line(image0, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
   
cv2.imshow('Traffic lines', image0)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('data\\output\\ref_res.png', image0)


