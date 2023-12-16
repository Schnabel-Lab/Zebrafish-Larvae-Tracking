# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def subfolderss(path):
  # Specify the path to the folder you want to list the subfolders of
  #folder_path = '/content/drive/MyDrive'
  folder_path = path
  # Use os.listdir() to get a list of all items (files and folders) in the specified directory
  all_items = os.listdir(folder_path)

  # Use a list comprehension to filter out only the subfolders (directories)
  subfolders = [item for item in all_items if os.path.isdir(os.path.join(folder_path, item))]

  # Now, the 'subfolders' list contains the names of all subfolders in the specified directory
  #print("Subfolders in the directory:", subfolders)
  return subfolders

scale_distance=52
scale_frames=30
patt='/content/drive/MyDrive/MCO/Jonathan/Experimento'

patt='/content/drive/MyDrive/MCO/Jonathan/Articulo_etanol/Experimentos'
bases='/content/drive/MyDrive/MCO/Jonathan/Articulo_etanol/Databases'
gra='/content/drive/MyDrive/MCO/Jonathan/Articulo_etanol/Graficos'
g_tratamiento=[]
g_muestra=[]
g_distance=[]
g_speed=[]
g_seg=[]
g_frames=[]
tot_dis=[]
dis_cent=[]
usar=[]
filess=['Tracks.csv','Spots.csv']
fram=15

subs1=subfolderss(patt)
for i in subs1:
  sec_patt=patt+'/'+i
  #print(sec_patt)
  subs2=subfolderss(sec_patt)
  if len(subs2):
    #print(subs2)
    for j in subs2:
      thir_patt=sec_patt+'/'+j
      #print(thir_patt)
      for k in filess:
        #print(i,j,k)
        if k==filess[0]:
          ruta=thir_patt+'/'+k
          df=pd.read_csv(ruta)
          #print(df.keys())
          g_tratamiento+=[i]
          g_muestra+=[j]
          aa=list(df.TOTAL_DISTANCE_TRAVELED)
          bb=float(aa[-1])/scale_distance
          g_distance+=[bb]
          aa=list(df.TRACK_MEAN_SPEED)
          bb=float(aa[-1])*scale_frames/scale_distance
          g_speed+=[bb]
          aa=list(df.TRACK_STOP)
          bb=float(aa[-1])/scale_frames
          g_seg+=[bb]
          bb=float(aa[-1])
          g_frames+=[bb]
          #print('done')
        else:
          ruta=thir_patt+'/'+k
          df0=pd.read_csv(ruta)
          frames=list(df0.FRAME)
          #print(len(frames))
          posX=list(df0.POSITION_X)
          posY=list(df0.POSITION_Y)
          reducido={'FRAME':frames[3:],'POSITION_X':posX[3:],'POSITION_Y':posY[3:]}
          df=pd.DataFrame(reducido)
          df['FRAME'] = df['FRAME'].astype(float)
          df['POSITION_X'] = df['POSITION_X'].astype(float)/scale_distance
          df['POSITION_Y'] = df['POSITION_Y'].astype(float)/scale_distance
          df=df.sort_values(by='FRAME')
          #print(list(df.FRAME))
          df['NORM_POS_X'] = df['POSITION_X']-df['POSITION_X'][0]
          aa=list(df.NORM_POS_X)
          #print(aa)
          bb=aa[1:]+[aa[0]]
          df['ShiftX']=bb
          df['NORM_POS_Y'] = df['POSITION_Y']-df['POSITION_Y'][0]
          aa=list(df.NORM_POS_Y)
          #print(aa)
          bb=aa[1:]+[aa[0]]
          df['ShiftY']=bb
          df['AbsX']=df.ShiftX-df.NORM_POS_X
          df['AbsX']=df.AbsX.abs()
          df['AbsY']=df.ShiftY-df.NORM_POS_Y
          df['AbsY']=df.AbsY.abs()
          df['Hypotenuse'] = np.sqrt(df['AbsY']**2 + df['AbsX']**2)
          aa=list(df.Hypotenuse)
          fram=15
          if len(aa)>fram:
            usar+=[1]
          else:
            usar+=[0]
          #print(aa)
          aa=aa[:fram]
          #print(aa)
          tot_dis+=[sum(aa)]
          aax=list(df.NORM_POS_X)
          aay=list(df.NORM_POS_Y)
          aax=aax[-1]
          aay=aay[-1]
          hyp=np.sqrt(aax**2 + aay**2)
          dis_cent+=[hyp]

          """
          aa=list(df[df.FRAME==10.0]['FRAME'])
          bb=float(aa[-1])/scale_frames
          g_seg+=[bb]
          df['']
          """
          f, ax = plt.subplots(figsize=(5, 5))
          plt.plot(df.NORM_POS_X,df.NORM_POS_Y)
          #plt.plot(df['NORM_POS_X'][-1],df['NORM_POS_Y'][-1],'*g')
          plt.plot(df['NORM_POS_X'][0],df['NORM_POS_Y'][0],'or')
          plt.ylim(-14,14)
          plt.xlim(-14,14)
          #plt.show()
          plt.savefig(thir_patt+'/tracking.png',dpi=300)
          plt.clf()
          #df=df.sort_values(by='FRAMES')
