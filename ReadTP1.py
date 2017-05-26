# -*- coding: utf-8 -*-
"""
Created on Sun May 21 18:15:00 2017

@author: bruno
"""
import numpy as np
import pandas as pd
    


def ajustadf(df):
    ret=pd.DataFrame()
    df["mkt"]=df.mean(axis=1)
    for column in df:
        ret[column]=np.log(df[column].shift(1)/df[column])
    ret=ret.dropna(axis=0)

    reversed_ret = ret.iloc[::-1]
    dates=reversed_ret.index
    return reversed_ret

def ReadCsv(archivo):
    df=pd.read_csv(archivo,index_col="Time",sep="\t",decimal=","
                   ,parse_dates=True)

    ret=ajustadf(df)
    return ret
    

