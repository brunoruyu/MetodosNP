# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:14:40 2017

@author: bruno
"""
import numpy as np
import Models
from scipy.stats import rankdata

def ConstRetT0(estimW,eventW,L1,L2,Activos):
    mu=np.mean(estimW,axis=0)
    ARestim=estimW-mu
    AR0=eventW[0]-mu
    sAi=np.sqrt((1/(L1-2))*np.sum(ARestim**2,axis=0))
    AR0p=AR0/sAi
    T=np.sum(AR0p)/np.sqrt(Activos)
    return T
    
def MktRetT0(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,Activos):
    sAi=np.sqrt((1/(L1-2))*np.sum(ARestim**2,axis=0))
    AR0p=AR[0]/sAi
    T=np.sum(AR0p)/np.sqrt(Activos)
    return T
 
def SignoT0(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,L2,Activos):
    S=[]
    S= [ [ 1 if ARestim.item(j,i) > 0  else 0 for i in range(0,Activos) ] 
     for j in range(0, L1) ]
#    print(ARestim,"\n",S)
    p=np.sum(S)/(L1*Activos)
    CAR0=AR[0]
    w= (CAR0 > 0).sum()
    T=(w-Activos*p)/np.sqrt(Activos*p*(1-p))
    return T    

    
def RangoT0(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,L2,Activos):
    A=np.concatenate((ARestim,AR),axis=0)
    K=A.copy()
    for i in range(0,Activos):  
        #order = A[:,i].argsort()
        #print("i ",i,A[:,i],order)
        K[:,i] =rankdata(A[:,i], method='ordinal') #order #.argsort()

    U=K/(L1+L2+1)
    V=U-0.5
    b=np.sum(V,axis=1)**2
    f=1.0/(Activos*(L1+L2))
    sU=np.sqrt(f*np.sum(b))   
    T=(np.sum(V[L1]))/(np.sqrt(Activos)*sU)

    return T
    
def CallModelT0(estimW,eventW,estimMkt,eventMkt,Activos):
    L1=len(estimW)
    L2=len(eventW)
    (ARestim,AR)=Models.Calc_AR(estimW,eventW,estimMkt,eventMkt,L1,L2,Activos) 
#    print("Calc_AR listo")
    
    TCR=ConstRetT0(estimW,eventW,L1,L2,Activos)
#    print("ConstRet listo")
    TMR=MktRetT0(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,Activos)
#    print("MktRet listo")
    TSig=SignoT0(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,L2,Activos)
#    print("Signo listo")
    TRan=RangoT0(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,L2,Activos)    
#    print("Rango listo")
    return TCR,TMR,TSig,TRan    
        
        
