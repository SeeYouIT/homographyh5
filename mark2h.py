import numpy as np
import cv2
import random


num = '000005'
head = './pdata/Y'+num
out_txt = './pdata/'+num + '.txt'
data_txt = head + '.txt'
allline = ''
with open(data_txt, 'r') as f:
    line1 = f.readline()
    line1 = line1.split(' ')
    line2 = f.readline()
    line2 = line2.split(' ')

    w = 1280
    h = 720
    pts_src = np.array([[int(line1[0].split(',')[0])/w*129, int(line1[0].split(',')[1])/h*129],[int(line1[1].split(',')[0])/w*129, int(line1[1].split(',')[1])/h*129],[int(line1[2].split(',')[0])/w*129, int(line1[2].split(',')[1])/h*129],[int(line1[3].split(',')[0])/w*129,int(line1[3].split(',')[1])/h*129]])
    pts_dst = np.array([[int(line2[0].split(',')[0])/w*129, int(line2[0].split(',')[1])/h*129],[int(line2[1].split(',')[0])/w*129, int(line2[1].split(',')[1])/h*129],[int(line2[2].split(',')[0])/w*129, int(line2[2].split(',')[1])/h*129],[int(line2[3].split(',')[0])/w*129,int(line2[3].split(',')[1])/h*129]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    s = h
    print(h)

    hstr = str(h[0,0])+','+str(h[0,1])+','+str(h[0,2])+','+str(h[1,0])+','+str(h[1,1])+','+str(h[1,2])+','+str(h[2,0])+','+str(h[2,1])

    pts_dst = np.array([[int(line1[0].split(',')[0]), int(line1[0].split(',')[1])],[int(line1[1].split(',')[0]), int(line1[1].split(',')[1])],[int(line1[2].split(',')[0]), int(line1[2].split(',')[1])],[int(line1[3].split(',')[0]),int(line1[3].split(',')[1])]])

    pts_src = np.array([[int(line2[0].split(',')[0]), int(line2[0].split(',')[1])],[int(line2[1].split(',')[0]), int(line2[1].split(',')[1])],[int(line2[2].split(',')[0]), int(line2[2].split(',')[1])],[int(line2[3].split(',')[0]),int(line2[3].split(',')[1])]])
    

    h, status = cv2.findHomography(pts_src, pts_dst)
    print(h)

    fstr = str(h[0,0])+','+str(h[0,1])+','+str(h[0,2])+','+str(h[1,0])+','+str(h[1,1])+','+str(h[1,2])+','+str(h[2,0])+','+str(h[2,1])


    file = open(out_txt,'w')
    file.write(hstr + ',' + fstr)
    file.close()

    print(np.dot(s, h))