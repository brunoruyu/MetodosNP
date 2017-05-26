# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:14:40 2017

@author: bruno
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

def linearfit (X,Y):
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    return results.params

def ConstRet(estimW,eventW,L1,L2):
    mu=np.mean(estimW,axis=0)
    ARestim=estimW-mu
    AR=eventW-mu
    se2=(1/(L1-2))*np.sum(ARestim**2,axis=0)
    #sigmae=np.std(estimW,axis=0) #este divide por L1, pero dividimos por L1-2
    sCAR2=(L2+1)*se2
    CAR=np.sum(AR,axis=0)
    sCAAR2=np.sum(sCAR2)
    CAAR=np.mean(CAR)
    T=CAAR/np.sqrt(sCAAR2)
    return T
    
def MktRet(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,L2):
    se2=(1/(L1-2))*np.sum(ARestim**2,axis=0)
#   sigmae=np.std(estimW,axis=0) #este divide por L1, pero dividimos por L1-2
    sCAR2=(L2+1)*se2
    CAR=np.sum(AR,axis=0)
    sCAAR2=np.sum(sCAR2)
    CAAR=np.mean(CAR)
    T=CAAR/np.sqrt(sCAAR2)
#    print(T)
    return T
 
def Signo(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,L2,Activos):
    S=[]
    S= [ [ 1 if ARestim.item(j,i) > 0  else 0 for i in range(0,Activos) ] 
     for j in range(0, L1) ]
#    print(ARestim,"\n",S)
    p=np.sum(S)/(L1*Activos)
    CAR=np.sum(AR,axis=0)
    w= (CAR > 0).sum()
    
    T=(w-Activos*p)/np.sqrt(Activos*p*(1-p))
    return T    

    
def Rango(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,L2,Activos):
    A=np.concatenate((ARestim,AR),axis=0)
    K=A.copy()
    for i in range(0,Activos):  #REVISAR EL ORDENAMIENTO
        order = A[:,i].argsort()
        #print("i ",i,A[:,i],order)
        K[:,i] = order   #.argsort()
    
    Lm=(L1+L2+1)/2.0    
    Kt=np.mean(K)
    Kd=np.mean(K[L1:,:])
    Z=np.sqrt(L2)*(Kd-Lm)/np.abs(Kt-Lm)
#    print('A',A)
#    print('K',K)
#    print(Lm,Kt,Kd,Z)
    return Z
    
def Calc_AR(estimW,eventW,estimMkt,eventMkt,L1,L2,Activos):
    ARestim=np.zeros((L1,Activos))
    AR=np.zeros((L2,Activos))
    for i in range(0,Activos):
        #print("i",i,estimMkt[:,i],estimW[:,i])
        Bet=linearfit(estimMkt[:,i],estimW[:,i])
        a=Bet[0]
        b=Bet[1]   
        ARestim[:,i]=estimW[:,i]-(a+b*estimMkt[:,i])
        AR[:,i]=eventW[:,i]-(a+b*eventMkt[:,i])
    
    return ARestim,AR

def CallModel(estimW,eventW,estimMkt,eventMkt,Activos):
    L1=len(estimW)
    L2=len(eventW)
    (ARestim,AR)=Calc_AR(estimW,eventW,estimMkt,eventMkt,L1,L2,Activos) 
    print("Calc_AR listo")
    
    TCR=ConstRet(estimW,eventW,L1,L2)
    print("ConstRet listo")
    TMR=MktRet(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,L2)
    print("MktRet listo")
    TSig=Signo(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,L2,Activos)
    print("Signo listo")
    TRan=Rango(estimW,eventW,estimMkt,eventMkt,ARestim,AR,L1,L2,Activos)    
    print("Rango listo")
    return TCR,TMR,TSig,TRan    
        
        
        
        
        
        
        
        
        
        
        
        # B= linearfit(TimeVect[k*n[i]:(k+1)*n[i]],Xt[k*n[i]:(k+1)*n[i],j])