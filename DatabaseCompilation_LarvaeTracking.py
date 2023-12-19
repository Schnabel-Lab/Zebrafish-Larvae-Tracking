# -*- coding: utf-8 -*-
#Organize Data
# Only for Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Asking scales
scale_distance=input('Pixels per milimeter: ')
scale_distance=int(scale_distance)
scale_frames=input('Frames per second: ')
scale_frames=int(scale_frames)

# Function for scanning
def subfolderss(path):
  # Specify the path to the folder you want to list the subfolders of
  folder_path = path
  # List of all items (files and folders) in the specified directory
  all_items = os.listdir(folder_path)
  # Filter out only the subfolders (directories)
  subfolders = [item for item in all_items if os.path.isdir(os.path.join(folder_path, item))]
  return subfolders

# Asking Folder for
path=input('Path where the folders of each condition: ')
bases=input('Path where you want to store the Database: ')
gra=input('Path where you want to store the Larvae trackings: ')

# Initial list for Database
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

# Scaning
subs1=subfolderss(patt)
for i in subs1:
  sec_patt=patt+'/'+i
  subs2=subfolderss(sec_patt)
  if len(subs2):
    for j in subs2:
      thir_patt=sec_patt+'/'+j
      for k in filess: # For Database
        if k==filess[0]:
          ruta=thir_patt+'/'+k
          df=pd.read_csv(ruta)
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
        else: # For individual larvae tracking
          ruta=thir_patt+'/'+k
          df0=pd.read_csv(ruta)
          frames=list(df0.POSITION_T)
          posX=list(df0.POSITION_X)
          posY=list(df0.POSITION_Y)
          reducido={'FRAME':frames[3:],'POSITION_X':posX[3:],'POSITION_Y':posY[3:]}
          df=pd.DataFrame(reducido)
          df['FRAME'] = df['FRAME'].astype(float)
          df=df.sort_values(by='FRAME')
          df['POSITION_X'] = df['POSITION_X'].astype(float)/scale_distance
          df['POSITION_Y'] = df['POSITION_Y'].astype(float)/scale_distance
          df['NORM_POS_X'] = df['POSITION_X']-df['POSITION_X'].iloc[0]
          aa=list(df.NORM_POS_X)
          bb=aa[1:]+[aa[0]]
          df['ShiftX']=bb
          df['NORM_POS_Y'] = df['POSITION_Y']-df['POSITION_Y'].iloc[0]
          aa=list(df.NORM_POS_Y)
          bb=aa[1:]+[aa[0]]
          df['ShiftY']=bb
          df['AbsX']=df.ShiftX-df.NORM_POS_X
          df['AbsX']=df.AbsX.abs()
          df['AbsY']=df.ShiftY-df.NORM_POS_Y
          df['AbsY']=df.AbsY.abs()
          df['Hypotenuse'] = np.sqrt(df['AbsY']**2 + df['AbsX']**2)
          aa=list(df.Hypotenuse)
          # You can change the frames depending how much time you want to test or filter
          fram=15
          if len(aa)>fram:
            usar+=[1]
          else:
            usar+=[0]
          aa=aa[:fram]
          tot_dis+=[sum(aa)]
          # Distance from center
          aax=list(df.NORM_POS_X)
          aay=list(df.NORM_POS_Y)
          aax=aax[-1]
          aay=aay[-1]
          hyp=np.sqrt(aax**2 + aay**2)
          dis_cent+=[hyp]
          # Plotting larvae tracking
          f, ax = plt.subplots(figsize=(5, 5))
          plt.plot(df.NORM_POS_X,df.NORM_POS_Y)
          plt.plot(0,0,'or')
          plt.ylim(-14,14)
          plt.xlim(-14,14)
          # Save each larvae tracking in the folder for Graphs with its own ID
          plt.savefig(gra+'/'+i+'_'+j+'tracking.png',dpi=300)
          plt.clf()

# Database creation
fin='Dis_in_'+str(fram)+'fra'
guardado={'Treatment':g_tratamiento,'Muestra':g_muestra,'Total_distance':g_distance,'Speed':g_speed,'Seconds':g_seg,fin:tot_dis,
          'Frames':g_frames,'For_use':usar,'From_center':dis_cent}
df=pd.DataFrame(guardado)

graficoss=list(df.keys())
graficoss=graficoss[2:]

# Plotting per experiment
plt.subplots(figsize=[15,5])
for i in graficoss:
  sns.boxplot(df, x="Treatment", y=i)
  # Add in points to show each observation
  sns.stripplot(df, x="Treatment", y=i, size=4, color=".3")
  plt.savefig(gra+'/'+i+'_all.png')
  plt.clf()

# Merge replicates, use your own labels
def trat(x):
  if x in ['Ctrl_Exp1_72', 'Ctrl_Exp2_72', 'Ctrl_Exp3_72']:
    return 'Ctrl_72hpf'
  elif x in ['EtOH_Exp3_72', 'EtOH_Exp1_72', 'EtOH_Exp2_72']:
    return 'EtOH_72hpf'

df['Treatment_complet']=df.Treatment
df['Treatment']=df.Treatment_complet.apply(trat)

df.keys()
# Store Database
df.to_csv(bases+'/Database.csv', index=False)

# Order in the graph
orden=['Ctrl_72hpf','EtOH_72hpf']
for i in graficoss:
  sns.boxplot(df, x="Tratamiento", y=i, order= orden)
  # Add in points to show each observation
  sns.stripplot(df, x="Tratamiento", y=i, size=4, color=".3",order=orden)
  plt.savefig(gra+'/'+i+'.png')
  plt.show()
  plt.clf()
