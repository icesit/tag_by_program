import numpy as np
import os
import sys

import json
import cv2
import numpy as np
import pylab as plt
import matplotlib.image as mplimg


IMG_SZ=384  #image size n*n
blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold
up_roi_hight = IMG_SZ
up_roi_x = IMG_SZ

# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 30
max_line_gap = 30

IMG_FILE_PATH='./photo'
IMG_FILE_OUT_PATH='./csv'

def gen_fname_list(rootdir='./'):
    flist=[]
    for pname,dnames,fnames in os.walk(rootdir):      
        flist +=[ os.path.join(pname,fname) for fname in fnames]
    return flist

def clean_lines(lines, threshold):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
      mean = np.mean(slope)
      diff = [abs(s - mean) for s in slope]
      idx = np.argmax(diff)
      if diff[idx] > threshold:
        slope.pop(idx)
        lines.pop(idx)
      else:
        break

def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)
  
    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))
  
    return [(xmin, ymin), (xmax, ymax)]

#draw vertical lanes
def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
    left_lines, right_lines = [], []
    if(lines is None):
        return
    for line in lines:
      for x1, y1, x2, y2 in line:
        devided = (x2 - x1)
        if(devided == 0):
            continue
        k = (y2 - y1) / devided
        if k < 0:
          left_lines.append(line)
        else:
          right_lines.append(line)
  
    if (len(left_lines) <= 0 or len(right_lines) <= 0):
      return img
  
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    left_points = [(x1, y1) for line in left_lines for x1,y1,x2,y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1,y1,x2,y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]

    #print(left_points)
    #print(left_points.shape)
    left_vtx = calc_lane_vertices(left_points, up_roi_hight, img.shape[0])
    right_vtx = calc_lane_vertices(right_points, up_roi_hight, img.shape[0])
  
    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
      for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lanes2(img, points, color=[255, 0, 0], thickness=2):
    print(points)
    if(len(points) < 2):
        return
    vtx = calc_lane_vertices(points, up_roi_hight, img.shape[0])
    cv2.line(img, vtx[0], vtx[1], color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lanes(line_img, lines)
    #draw_lines(line_img, lines)
    return line_img

def find_lane_by_hough(inputfilename):
    #read data
    fname = './photo/'+inputfilename.strip('\n')+'.jpeg'
    print('read data '+ fname)
    imgcv=cv2.imread(fname)
    imgcv=cv2.resize(imgcv,(IMG_SZ,IMG_SZ),interpolation=cv2.INTER_CUBIC)
    img=cv2.imread(fname).astype(np.uint8)
    img=cv2.resize(img,(IMG_SZ,IMG_SZ),interpolation=cv2.INTER_CUBIC)
    idx=max(fname.rfind('/'),fname.rfind('\\'))
    fname_raw=fname[idx+1:fname.rfind('.')]
    img_seg = cv2.imread('./resultseg/'+fname_raw+'.jpeg').astype(np.float32)
    fname_csv=IMG_FILE_OUT_PATH+'/seg_'+fname_raw+'.csv'
    type_id=np.genfromtxt(fname_csv, delimiter=',').astype(int).reshape(IMG_SZ,IMG_SZ)

    #mask and find edge
    gray = cv2.cvtColor(imgcv, cv2.COLOR_RGB2GRAY)
    #print(gray.shape)
    #make mask of roi n*n
    mask = np.zeros_like(gray)
    global up_roi_hight
    for y in range(IMG_SZ):
        for x in range(IMG_SZ):
            if(type_id[y,x] == 7):
                mask[y,x] = 255
                if(up_roi_hight == IMG_SZ):
                    up_roi_hight = y            
    kernel = np.ones((7,7),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 5)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    roi_gray = cv2.bitwise_and(blur_gray, mask)
    #print(roi_gray.depth())
    roi_edges = cv2.Canny(roi_gray, canny_lthreshold, canny_hthreshold)
    
    #find line
    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
    #res_img = np.zeros_like(img)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    up_roi_hight = IMG_SZ

    #img = cv2.bitwise_and(img, mask)
    plt.clf()
    plt.subplot(2,3,1)
    plt.imshow(img_seg/255.0)
    plt.subplot(2,3,2)
    plt.imshow(mask/255.0, cmap='gray')
    plt.subplot(2,3,3)
    plt.imshow(img/255.0)
    plt.subplot(2,3,4)
    plt.imshow(roi_edges/255.0, cmap='gray')
    plt.subplot(2,3,5)
    plt.imshow(res_img/255.0)
    #plt.pause(0.5)
    plt.show()

def find_lane_by_convexhull(inputfilename):
    #read data
    fname = './photo/'+inputfilename.strip('\n')+'.jpeg'
    print('read data '+ fname)
    imgcv=cv2.imread(fname)
    imgcv=cv2.resize(imgcv,(IMG_SZ,IMG_SZ),interpolation=cv2.INTER_CUBIC)
    img=cv2.imread(fname).astype(np.uint8)
    img=cv2.resize(img,(IMG_SZ,IMG_SZ),interpolation=cv2.INTER_CUBIC)
    idx=max(fname.rfind('/'),fname.rfind('\\'))
    fname_raw=fname[idx+1:fname.rfind('.')]
    img_seg = cv2.imread('./resultseg/'+fname_raw+'.jpeg').astype(np.float32)
    fname_csv=IMG_FILE_OUT_PATH+'/seg_'+fname_raw+'.csv'
    type_id=np.genfromtxt(fname_csv, delimiter=',').astype(int).reshape(IMG_SZ,IMG_SZ)

    gray = cv2.cvtColor(imgcv, cv2.COLOR_RGB2GRAY)
    #make mask of roi n*n
    mask = np.zeros_like(gray)
    global up_roi_hight
    global up_roi_x
    for y in range(IMG_SZ):
        for x in range(IMG_SZ):
            if(type_id[y,x] == 7):
                mask[y,x] = 255
                if(up_roi_hight == IMG_SZ):
                    up_roi_hight = y
                    up_roi_x = x

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 5)
    mask = cv2.erode(mask, kernel, iterations = 8)
    image, contours, hierarchy = cv2.findContours(mask, 3, 2)
    #find largest contour
    maxarea = 0
    maxarea_id = -1
    for i in range(len(contours)):
        tmparea = cv2.contourArea(contours[i])
        if(tmparea > maxarea):
            maxarea = tmparea
            maxarea_id = i

    if(maxarea_id >= 0):
        cnt = contours[maxarea_id]
        #approx = cv2.approxPolyDP(cnt, 20, True)
        approx = cv2.convexHull(cnt)
        #apart the points left and right
        left_points, right_points = [],[]
        left_points.append((up_roi_x, up_roi_hight))
        right_points.append((up_roi_x, up_roi_hight))
        flag_left_corner = False
        flag_right_corner = False
        for i in range(approx.shape[0]):
            if(approx[i,0,1] < IMG_SZ-3 or (approx[i,0,0] < IMG_SZ-3 and approx[i,0,0] > 3)):
                if(approx[i,0,0] > up_roi_x):
                    if(approx[i,0,0] > IMG_SZ-3 or approx[i,0,1] > IMG_SZ-3):
                        flag_right_corner = True
                    right_points.append((approx[i,0,0], approx[i,0,1]))
                else:
                    if(approx[i,0,0] < 3 or approx[i,0,1] > IMG_SZ-3):
                        flag_left_corner = True
                    left_points.append((approx[i,0,0], approx[i,0,1]))
        if(len(right_points)==1 or not flag_right_corner):
            right_points.append((IMG_SZ-1,IMG_SZ-1))
        if(len(left_points)==1 or not flag_left_corner):
            left_points.append((0,IMG_SZ-1))
        #print(left_points)
        #print(right_points)
        res_img = np.zeros(img.shape, np.uint8)
        res_img = img.copy()
        draw_lanes2(res_img, left_points)
        draw_lanes2(res_img, right_points)
        #print(right_points)
        #print(approx[1,0])
        #print(approx[0,0,0])
        cv2.polylines(img, [approx], True, (0, 255, 0), 2)

    up_roi_hight = IMG_SZ
    up_roi_x = IMG_SZ

    plt.clf()
    plt.subplot(1,4,1)
    plt.imshow(img_seg/255.0)
    plt.subplot(1,4,2)
    plt.imshow(mask/255.0, cmap='gray')
    plt.subplot(1,4,3)
    plt.imshow(img/255.0)
    plt.subplot(1,4,4)
    plt.imshow(res_img/255.0)
    #plt.pause(0.5)
    plt.show()

#start here
#flist=gen_fname_list(IMG_FILE_PATH)
f = open('fnames.txt', 'r')
flist = f.readlines()

for filename in flist:
    plt.figure(figsize=(15,6))
    #find_lane_by_hough(filename)
    find_lane_by_convexhull(filename)
