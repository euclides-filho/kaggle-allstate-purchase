"""
Project: http://www.kaggle.com/c/allstate-purchase-prediction-challenge
Ranking: 9th from 1571 teams
Work Period: 12-may-2014 to 19-may-2014
Author:  Euclides Fernandes Filho
email:   euclides5414@gmail.com
"""
ROOT = "./"
ROOT_SQL = ROOT + "sql/"

DB_PASSWORD = TO_DEFINE
DB = 'ALL_STATE'
HOST = '127.0.0.1'
USER = 'root'

import mysql.connector
from pre_parse import pre_parse
import pandas as pd
from time import time

def db_setup():    
    cnx = None
    sql_setup = ""
    sql_load_train = ""
    sql_load_test = ""
    sql_insert_train_trunc = ""
    with open(ROOT_SQL + 'setup.sql','r') as f:
        sql_setup = f.read()
    with open(ROOT_SQL + 'load_train.sql','r') as f:
        sql_load_train = f.read()
    with open(ROOT_SQL + 'load_test.sql','r') as f:
        sql_load_test = f.read()
    with open(ROOT_SQL + 'insert_train_trunc.sql','r') as f:
        sql_insert_train_trunc = f.read()
    
        
    try:
        cnx = mysql.connector.connect(user=USER, password=DB_PASSWORD,host=HOST,database=DB)
        cursor = cnx.cursor()
        t0 = time()
        
        cursor.execute(sql_setup,multi=True)
        print sql_setup
        print "::: EXECUTED ::: in %2.2f s" % (time() - t0)
        print
        
        cursor.execute(sql_load_train)
        cnx.commit()
        print sql_load_train
        print "::: EXECUTED ::: in %2.2f s" % (time() - t0)
        print
        
        cursor.execute(sql_load_test)
        cnx.commit()
        print sql_load_test
        print "::: EXECUTED ::: in %2.2f s" % (time() - t0)
        print
        TOTAL_SAMPLES = 9
        for i in range(1,TOTAL_SAMPLES + 1):
            sql = sql_insert_train_trunc % str(i)
            cursor.execute(sql)
            print sql
            print "::: EXECUTED ::: in %2.2f s" % (time() - t0)
            print
        cnx.commit()
        
    finally:
        if cnx is not None:
            cnx.close()
            cursor.close()
    

def main():
    print __doc__
    #pre_parse()
    db_setup()

if __name__ == '__main__':
    main()
