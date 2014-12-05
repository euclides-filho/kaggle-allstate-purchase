"""
Project: http://www.kaggle.com/c/allstate-purchase-prediction-challenge
Ranking: 9th from 1571 teams
Work Period: 12-may-2014 to 19-may-2014
Author:  Euclides Fernandes Filho
email:   euclides5414@gmail.com
"""
ROOT = "./"
ROOT_SQL = ROOT + "sql/"
ROOT_SUB = ROOT + "sub/"

DB_PASSWORD = TO_DEFINE
DB = 'ALL_STATE'
HOST = '127.0.0.1'
USER = 'root'

import mysql.connector
import numpy as np
import pandas as pd
from imports import *

ROOT = "./"
ROOT_SQL = ROOT + "sql/"
ROOT_DATA = ROOT + "data/"

cols=['customer_ID','shopping_pt','day','state','location','group_size','homeowner','car_age','car_value','risk_factor','age_oldest','age_youngest','married_couple','C_previous','duration_previous','A','B','C','D','E','F','G','cost']
cols_y = ['y_shopping_pt','y_day','y_state','y_location','y_group_size','y_homeowner','y_car_age','y_car_value','y_risk_factor','y_age_oldest','y_age_youngest','y_married_couple','y_C_previous','y_duration_previous','y_A','y_B','y_C','y_D','y_E','y_F','y_G','y_cost']

ID_COL = 'customer_ID'

def assertEqual(v1, v2, info=''):
    if v1 is None:
        isE = True
    elif len(v1) != len(v2):
        isE = False
    else:
        isE = True
        for i in range(len(v1)):
            isE = isE & (v1[i] == v2[i])
    if not isE:
        print "ASSERTION FAILED %s" % info
    return isE   


def parse(sample_num=1):
    sql_train_sn = ""
    sql_test = ""
    
    with open(ROOT_SQL + 'select_last_test.sql','r') as f:
        sql_test = f.read()
    with open(ROOT_SQL + 'select_last_train_sn.sql','r') as f:
        sql_train_sn = f.read()        
    
    DF_train, DF_test = None, None
    df_train, df_test = None, None
    y_train = None
    Y_train = None
    cnx = None

    try:
        cnx = mysql.connector.connect(user=USER, password=DB_PASSWORD,host=HOST,database=DB)
        for i, quote_num in enumerate([0,1]):
            sql_train = sql_train_sn % (str(sample_num),str(quote_num))
            file_name_train = ROOT_DATA + "TRAIN_LAST_SN=%i_SHPT=%i.csv" % (sample_num,quote_num)
            file_name_test = ROOT_DATA + "TEST_LAST_SHPT=%i.csv" % (quote_num)

            if not path.exists(file_name_train):
                tdf_train = pd.io.sql.read_frame(sql_train, cnx)
                tdf_train.to_csv(file_name_train,sep=",")
                print '\t\t\tFILE: "%s" SAVED' % (file_name_train) 
            else:
                tdf_train = pd.read_csv(file_name_train)
                print '\t\t\tFILE: "%s" LOADED' % (file_name_train) 
                
            if not path.exists(file_name_test):
                tdf_test = pd.io.sql.read_frame(sql_test, cnx)
                tdf_test.to_csv(file_name_test,sep=",")
                print '\t\t\tFILE: "%s" SAVED' % (file_name_test) 
            else:
                tdf_test = pd.read_csv(file_name_test)
                print '\t\t\tFILE: "%s" LOADED' % (file_name_test) 
            
            if not i:                
                IDs = tdf_train[ID_COL].values
                y_train = tdf_train[cols_y]
                df_train = tdf_train[cols]
                df_test = tdf_test[cols]
            else:
                for col in cols:
                    if col in set(['customer_ID','time']):
                        continue
                    ncol = "%s_%i" % (col,quote_num)
                    df_train[ncol] = tdf_train[col]
                    df_test[ncol] = tdf_test[col]

            
        if DF_train is None:
            DF_train = df_train
            DF_test = df_test
            Y_train = y_train
       
        X, X_pred, ys = DF_train.convert_objects(convert_numeric=True), DF_test.convert_objects(convert_numeric=True), Y_train.convert_objects(convert_numeric=True)
        print X.shape, X_pred.shape, ys.shape
    finally:
        if cnx is not None:
            cnx.close()
    
    return X, X_pred, ys

    
if __name__ == '__main__':
    parse()

   


