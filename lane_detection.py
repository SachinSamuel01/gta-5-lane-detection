import numpy as np
import cv2
from sys import exit
import os
from math import sqrt
from statistics import mean

def length(list):
    p=list[0][0]-list[1][0]
    p=p*p
    q=list[0][1]-list[1][1]
    q=q*q
    p=sqrt(p+q)
    return p




def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 



def draw_st_lines_one(img,lines,h,w):
        
    

    try:
        
        line_dict={}
        lane_=[]
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            dicts={y1:x1,y2:x2}
            
            ttt=0
            
            min_y=min(y1,y2)
            max_y=max(y1,y2)

            x1=dicts[min_y]
            x2=dicts[max_y]
            y1=min_y
            y2=max_y

            x=(x1,x2)
            y=(y1,y2)
            A=np.vstack([x,np.ones(len(x))])
            m,y_=np.linalg.lstsq(A, y, rcond=None)[0]
            
            if len(line_dict)==0:
                line_dict[m]=[y_,[x1,y1],[x2,y2]]
                continue
                
            b=list(line_dict.keys())
            for i in b:
                
                intercept=line_dict[i][0]
                if abs(i)*1.2 > abs(m) >abs(i)*0.8:
                    if abs(intercept)*1.2>abs(y_)>abs(intercept)*0.8:
                        coord_1=line_dict[i][1]
                        coord_2=line_dict[i][2]
                        x1,y1,x2,y2=(coord_1[0]+x1)/2,(coord_1[1]+y1)/2,(coord_2[0]+x2)/2,(coord_2[1]+y2)/2
                        
                        del line_dict[i]
                        
                        
                        x=(x1,x2)
                        y=(y1,y2)
                        A=np.vstack([x,np.ones(len(x))])
                        m,y_=np.linalg.lstsq(A, y, rcond=None)[0]
                        line_dict[m]=[y_,[x1,y1],[x2,y2]]
                        ttt=1
                        break
            if ttt==0:
                line_dict[m]=[y_,[x1,y1],[x2,y2]]
        #print(line_dict)
        
        
        
        for j in list(line_dict.keys()):
            
            x1,y1,x2,y2=int(line_dict[j][1][0]),int(line_dict[j][1][1]),int(line_dict[j][2][0]),int(line_dict[j][2][1])

            coords1=(x1,y1)
            coords2=(x2,y2)
            line_s=[coords1,coords2]
            if len(lane_)==0:
                if y2-y1>=50 :
                    if x2>=500 or x2<=300:
                        lane_.append(line_s)
                
            elif len(lane_)==1:
                if y2-y1>=50 :
                    if x2>=500 or x2<=300:
                        if (lane_[0][1][0]>=500 and x2<=300) or (lane_[0][1][0]<=300 and x2>=500):
                            if length(line_s)>length(lane_[0]):
                                temp=lane_[0]
                                lane_[0]=line_s

                                lane_.append(temp)
                            else:
                                lane_.append(line_s)
                
                
            else:
                if y2-y1>=50 :
                    if x2>=500 or x2<=300:
                        
                        if length(line_s)>length(lane_[0]):
                            if (lane_[0][1][0]>=500 and x2<=300) or (lane_[0][1][0]<=300 and x2>=500):
                                temp=lane_[0]
                                lane_[0]=line_s

                                lane_[1]=temp
                        elif length(line_s)>length(lane_[1]) :
                             if (lane_[0][1][0]>=500 and x2<=300) or (lane_[0][1][0]<=300 and x2>=500):
                                lane_[1]=line_s
            #cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
        for i in lane_:
            x1,y1,x2,y2=i[0][0],i[0][1],i[1][0],i[1][1]
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),10)




    except:
        pass
    



            
        

        
    
    
def draw_st_lines_two(img,lines,height,width):
    
    
    try:
        ys = []  
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = np.vstack([x_coords,np.ones(len(x_coords))]).T
                m, b = np.linalg.lstsq(A, y_coords)[0]

                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        
        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])
        
        cv2.line(img,(l1_x1, l1_y1),(l1_x2, l1_y2),(255,0,0),10)
        cv2.line(img,(l2_x1, l2_y1),(l2_x2, l2_y2),(255,0,0),10)

    except:
        pass



      
    
    

def roi(img,v):
    mask=np.zeros_like(img)
    cv2.fillPoly(mask,[v],(255,0,0))
    masked =cv2.bitwise_and(img,mask)
    return masked

def lane_detection_processed_img(img,img2,h,w):
    #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.GaussianBlur(img,(11,11),0)
    img=cv2.Canny(img,150,210)
    #v=np.array([[0,0],[w,0],[w,h],[700,h],[700,350],[390,190],[70,310],[70,h],[0,h]])
    #v1=np.array([[700,h],[700,370],[380,210],[70,330],[70,h]])
    #v2=np.array([[50,200],[w-50,200],[w-50,h],[50,h]])


    #img=roi(img,v2)
    img=cv2.GaussianBlur(img,(5,5),0)
    
    lines=cv2.HoughLinesP(img,2,np.pi/180,600,np.array([]),150,5)
    #sobelx = np.absolute(cv2.Sobel(img,cv2.CV_64F,1,0,(3,3)))
    #print(sobelx)
    draw_st_lines_one(img2,lines,h,w)
    return img,img2
    


def a():
    d=r'C:\Users\samue\OneDrive\Desktop\Newfolder'
    os.chdir(d)
    while(True):
        img=cv2.imread('k1.jpg')
        cv2.line(img,(20,20),(200,200),(255,0,0),3)
        cv2.imshow('test2',img)

        if cv2.waitKey(1) & 0xFF==ord('q'):
                cv2.destroyAllWindows()
                break
            
    


def b():
    
    p=4
    p=p*p
    
    q=3
    q=q*q
    p=sqrt(p+q)
    print(p)

#b()
#a()