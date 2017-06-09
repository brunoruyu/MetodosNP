# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:33:44 2017

@author: bruno
"""
import numpy as np
import pandas as pd
import Models
import ModelsT0
import ReadTP1
import statsmodels.api as sm

#Voy a armar 1000 experimentos
#Cada Experimento tiene 100 Activos elegidos al azar
#Una vez elegido cada Activo elijo una fecha al azar para que sea el
#inicio de los 260 días (es decir la fecha no puede estar en los últimos 260 d)
#entonces va de 0 a 2256 en el índice de fechas

#Las Keys van a ser los experimentos de 1 a 1000
#los values van a ser una tupla con (activo,fechainicio)
#Activo va a ser un int de 0 a 1077 -- ret.iloc[:,activo])
#La fechainicio va a ser el int del índice de ret ---- ret.iloc[3,1]

#Si no funciona, uso una matriz de 3 dimensiones y listo

#print(np.array(ret).item(2,1)) (tiempo,Activo)
archivo="dataTP1.dat"
maxAct,maxTime=1077,2256
Samples,Activos,L1,L2=1000,150,250,10
lamb,eta=0.1,0.5
if(False): #True para Test
    archivo="testTP1.dat"
    maxAct,maxTime=3,7
    Samples,Activos,L1,L2=1,3,4,2
   
retdf=ReadTP1.ReadCsv(archivo) #ya vienen ordenados de pasado a futuro
ret=np.array(retdf)
print("Finished Reading")

v=[]
v= [ np.std(ret[:,j],ddof=1)  for j in range(0, Activos) ]

design=np.zeros((2,Activos))
estimW=np.zeros((L1,Activos))
eventW=np.zeros((L2,Activos))
estimMkt=np.zeros((L1,Activos))
eventMkt=np.zeros((L2,Activos))
T=np.zeros((Samples,4))
T0=np.zeros((Samples,4))
#print(exp)

for s in range(0,Samples):
    #print(s)
    for i in range(0,Activos):
        
        act=np.random.randint(1,high=maxAct) #int(design.item(0,i))
        tinic=np.random.randint(1,high=maxTime) #int(design.item(1,i))
#        print(i,act,tinic)
        
        estimW[:,i]=np.copy(ret[tinic:tinic+L1,act])
        estimMkt[:,i]=np.copy(ret[tinic:tinic+L1,-1])
        eventW[:,i]=np.copy(ret[tinic+L1:tinic+L1+L2,act])
        eventMkt[:,i]=np.copy(ret[tinic+L1:tinic+L1+L2,-1])
#    print("Eventos Armados")       
    
    T[s]=Models.CallModel(estimW,eventW,estimMkt,eventMkt,Activos)
    T0[s]=ModelsT0.CallModelT0(estimW,eventW,estimMkt,eventMkt,Activos)
    
    #el orden de T es Const,Mkt,Signo,Rango

w= (np.abs(T) > 1.96).sum(axis=0)
w=w/Samples
w0= (np.abs(T0) > 1.96).sum(axis=0)
w0=w0/Samples
print("w",w)
print("w0",w0)
np.savetxt('Out_100_10.dat', T, delimiter='\t')
np.savetxt('Out_100_0.dat', T0, delimiter='\t')


"""        
        for j in range(0,L1):    
           estimW[j,i]=np.array(ret).item(tinic+j,act)
           estimMkt[j,i]=np.array(ret).item(tinic+j,-1)
           #print("estim",i,j,tinic+j,act,estimMkt[j,i])
        for j in range(0,L2):             
           eventW[j,i]=np.array(ret).item(tinic+j+L1,act)
           eventMkt[j,i]=np.array(ret).item(tinic+j+L1,-1)
           #print("event",i,j,tinic+L1+j,act,eventMkt[j,i])
"""