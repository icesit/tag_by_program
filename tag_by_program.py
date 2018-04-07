# coding=utf-8

import numpy as np
import os
import sys
import codecs

import json
import cv2
import numpy as np
import pylab as plt
import matplotlib.image as mplimg

###########
# this part is for lane line extraction
# assume that road is parallel to direction
###########

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
    if(len(points) < 2):
        return
    vtx = calc_lane_vertices(points, up_roi_hight, img.shape[0])
    cv2.line(img, vtx[0], vtx[1], color, thickness)
    return vtx


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

    return_lines = []
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
        return_lines.append(draw_lanes2(res_img, left_points))
        return_lines.append(draw_lanes2(res_img, right_points))
        #print(right_points)
        #print(approx[1,0])
        #print(approx[0,0,0])
        cv2.polylines(img, [approx], True, (0, 255, 0), 2)

    up_roi_hight = IMG_SZ
    up_roi_x = IMG_SZ
    #print(return_lines)
    '''
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
    '''
    return return_lines, res_img, type_id

###########
# this part is for area segment
# assume that road is parallel to direction
###########

#input four points, for parallel road
def find_all_interested_line_parallel(lines):
    xl0 = lines[0][0][0]
    yl0 = lines[0][0][1]
    xl1 = lines[0][1][0]
    yl1 = lines[0][1][1]
    xr0 = lines[1][0][0]
    yr0 = lines[1][0][1]
    xr1 = lines[1][1][0]
    yr1 = lines[1][1][1]
    #x-x0 = k(y-y0)--> x=k*y + b
    #k=(x0-x1)/(y0-y1), because parallel road will has no y0==y1
    #b=(x0-k*y0)
    if(yl0 != yl1):
        kl01 = float(xl0-xl1) / (yl0-yl1)
        kr01 = float(xr0-xr1) / (yr0-yr1)
        bl01 = xl0 - kl01*yl0
        br01 = xr0 - kr01*yr0
        if(kl01 == kr01):
            print('[find intersect point]wrong input lines, paralell lines')
        else:
            yvan = int((br01 - bl01) / (kl01 - kr01))
            xvan = int(kl01*(yvan - yl0) + xl0)

            y_nearmid = int((IMG_SZ + yvan) / 2)
            y_midfar = int((y_nearmid + yvan) / 2)

            x_nearmid_l01 = int(kl01*(y_nearmid - yl0) + xl0)
            x_nearmid_r01 = int(kr01*(y_nearmid - yr0) + xr0)

            x_nearmid_l12 = int(x_nearmid_l01 - 3 / 7.5 * (x_nearmid_r01 - x_nearmid_l01))
            x_nearmid_r12 = int(x_nearmid_r01 + 3 / 7.5 * (x_nearmid_r01 - x_nearmid_l01))

            kl12 = float(xvan - x_nearmid_l12) / (yvan - y_nearmid)
            bl12 = xvan - kl12*yvan

            kr12 = float(xvan - x_nearmid_r12) / (yvan - y_nearmid)
            br12 = xvan - kr12*yvan

            xm00 = (x_nearmid_l01 + x_nearmid_r01) / 2
            km00 = float(xvan - xm00) / (yvan - y_nearmid)
            bm00 = xvan - km00*yvan

    #print((xvan, yvan), xm00, x_nearmid_l12, x_nearmid_r12, x_nearmid_l01, x_nearmid_r01)

    return (xvan, yvan), (km00, bm00), (kl01, bl01), (kr01, br01), (kl12, bl12), (kr12, br12), y_nearmid, y_midfar

##################
#________________#
#________/\______#
#_______/  \_____#
#      /    \    #
#     /      \   #
##################

def devide_region_parallel(lanelines, res_img):
    #find vanishing point
    #point_vanish, kbmid, kbl12, kbr12, y_nearmid, y_midfar = find_all_interested_point_parallel(lanelines)
    #print(point_vanish)
    return find_all_interested_line_parallel(lanelines)

def whether_obj_exit_in_region(region_now, new_detected_obj):
    new_obj = str(new_detected_obj)
    for obj in region_now:
        if(obj == new_obj):
            return True
    return False

def get_lnglat_from_filename(filename):
    a,b,c,d = filename.split('_')
    return b.replace('lng', ''), c.replace('lat', '')

def generate_json_parallel(filename, type_id_csv, type_dict, vanish_point, kbmid, kbl01, kbr01, kbl12, kbr12, y_nearmid, y_midfar):
    # go through all grids in image and add to list
    L_region = [[[],[],[]], [[],[],[]], [[],[],[]]]
    R_region = [[[],[],[]], [[],[],[]], [[],[],[]]]
    a=0
    b=0
    for y in range(vanish_point[1], IMG_SZ):
        for x in range(IMG_SZ):
            #judge how far from me in front near1 mid2 far3
            if(y < y_midfar):
                #far
                a = 3
            elif(y < y_nearmid):
                #mid
                a = 2
            else:
                #near
                a = 1
            # judge left right
            if((kbmid[0]*y + kbmid[1]) > x):
                #left
                #judge judge how far from road in0 near1 far2
                if((kbl12[0]*y + kbl12[1]) > x):
                    #far
                    b = 2
                elif((kbl01[0]*y + kbl01[1]) > x):
                    # near
                    b = 1
                else:
                    #in
                    b = 0
                #Lba
                if(not whether_obj_exit_in_region(L_region[b][a-1], type_id_csv[y,x])):
                    #print(type(type_id_csv[y,x]))
                    L_region[b][a-1].append(str(type_id_csv[y,x]))
            else:
                #right
                #judge judge how far from road in0 near1 far2
                if((kbr12[0]*y + kbr12[1]) < x):
                    #far
                    b = 2
                elif((kbr01[0]*y + kbr01[1]) < x):
                    # near
                    b = 1
                else:
                    #in
                    b = 0
                #Rba
                if(not whether_obj_exit_in_region(R_region[b][a-1], type_id_csv[y,x])):
                    R_region[b][a-1].append(str(type_id_csv[y,x]))
    #print(L_region)
    #write as json
    region_json = {}
    region_json['name'] = filename.strip('\n')+'.jpeg'
    region_json['num_of_road'] = '1'
    lng, lat = get_lnglat_from_filename(filename)
    region_json['lng'] = lng
    region_json['lat'] = lat
    road_json = {}
    road_json['number'] = '0'
    road_json['status'] = '在路中'
    road_json['relation'] = '0'
    road_json['passable'] = 'f'
    obj_json = {}
    obj_list = []
    for da in range(1, 4):
        for db in range(3):
            for i in L_region[db][da-1]:
                if i in dictionary.keys():
                    obj_json['seen'] = dictionary[i]
                    obj_list.append(obj_json)
                    obj_json = {}
            road_json['L'+str(db)+str(da)] = obj_list
            #print(obj_list)
            obj_list = []
            for i in R_region[db][da-1]:
                if i in dictionary.keys():
                    obj_json['seen'] = dictionary[i]
                    obj_list.append(obj_json)
                    obj_json = {}
            road_json['R'+str(db)+str(da)] = obj_list
            obj_list = []
    #print(road_json)
    road_list = []
    road_list.append(road_json)
    region_json['road'] = road_list
    return region_json


###########
#start here
###########
#flist=gen_fname_list(IMG_FILE_PATH)
## read list of image that need to be tag(starts with 'photo'
## and end with 'image0', without '.jpeg')
f = open('fnames.txt', 'r')
flist = f.readlines()

## read dictionary of type
fin = open('type.txt','r')
fdlist = fin.readlines()
dictionary = {}
for ln in fdlist:
    k, v = ln.split('.')
    dictionary[k] = v.strip('\n')
#print(dictionary['7'])

## open josn to write
fjson = codecs.open('auto_tag.json', 'w')
out_json = []

for filename in flist:
    plt.figure(figsize=(15,6))
    #find_lane_by_hough(filename)
    lanelines, res_img, type_id_csv = find_lane_by_convexhull(filename)

    vanish_point, kbmid, kbl01, kbr01, kbl12, kbr12, y_nearmid, y_midfar = devide_region_parallel(lanelines, res_img)
    #print(kbmid, kbl01, kbr01, kbl12, kbr12, y_nearmid, y_midfar)

    out_json.append(generate_json_parallel(filename, type_id_csv, dictionary, vanish_point, kbmid, kbl01, kbr01, kbl12, kbr12, y_nearmid, y_midfar))

#json.dump(out_json, fjson)
fjson.write(json.dumps(out_json, ensure_ascii=False))
f.close()
fjson.close()
fin.close()
