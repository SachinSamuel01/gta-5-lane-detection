from operator import iadd, iand
from re import I
import win32gui,win32ui,win32con
import numpy as np
import cv2
import time
from lane_detection import lane_detection_processed_img
import os


class model():
    h=0
    w=0
    a=180
    hwnd=None
    def __init__(self):
        #self.a=150
        self.hwnd = win32gui.FindWindow(None, 'Grand Theft Auto V')
        win_broder=win32gui.GetWindowRect(self.hwnd)
        self.w=win_broder[2]-win_broder[0]-10
        self.h=win_broder[3]-win_broder[1]-self.a-5
        #print(self.h,"::",self.w)
        
        
    def screen_grab(self):
        
        #bmpfilenamename = "out.bmp" #set this
        #hwnd=None
    
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj=win32ui.CreateDCFromHandle(wDC)
        cDC=dcObj.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(bmp)
        cDC.BitBlt((0,0),(self.w, self.h) , dcObj, (5,self.a), win32con.SRCCOPY)
        #dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h,self.w,4)
        #img=img[...,:3]
        img=np.ascontiguousarray(img)
        # Free Resources

        

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(bmp.GetHandle())
        return img

    
q=model()

def save_img():
    d=r'C:\Users\samue\OneDrive\Desktop\Newfolder\test1'
    os.chdir(d)
    f='test1.jpg'
    f1='test11.jpg'
    for i in range(200):
        img=q.screen_grab()
        img2=img
        img=cv2.GaussianBlur(img,(9,9),0)
        img=cv2.Canny(img,150,200)
        cv2.imshow('test2',img)
        if i==199:
            cv2.imwrite(f,img)
            cv2.imwrite(f1,img2)
            break
        
        else:
            pass
            
        print(i)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            break


#save_img()

def test1():
    while(True):
        begin=time.time()
        img=q.screen_grab()
        img2=img
        img,img2=lane_detection_processed_img(img,img2,q.h,q.w)
        
        cv2.imshow('test1',img2)
        cv2.imshow('test2',img)

        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            break
        print(1/(time.time()-begin))

test1()


def reinforment_learning():
    discrete_os_size=20
    pass


