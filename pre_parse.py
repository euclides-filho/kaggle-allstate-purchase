"""
Project: http://www.kaggle.com/c/allstate-purchase-prediction-challenge
Ranking: 9th from 1571 teams
Work Period: 12-may-2014 to 19-may-2014
Author:  Euclides Fernandes Filho
email:   euclides5414@gmail.com
"""
import numpy as np
import pandas as pd
from os import path
import conv
from time import sleep, time

from imports import *

ROOT = "./"
PRE_TRAIN_FILE = ROOT + 'data/train.csv'
PRE_TEST_FILE = ROOT + 'data/test_v2.csv'

TRAIN_FILE = ROOT + 'data/train_P.csv'
TEST_FILE = ROOT + 'data/test_v2_P.csv'


def pre_parse():
    convs = {'car_value':conv.conv_car_value, 'state':conv.conv_state, 'C_previous':conv.conv_C_previous, 'duration_previous':conv.conv_duration_previous, 'time':conv.conv_time}


    if not path.exists(TRAIN_FILE):
        train = pd.read_csv(PRE_TRAIN_FILE, converters=convs)
        train = do_risk(train)
        
        train.to_csv(TRAIN_FILE, sep=',',na_rep="NA")
    else:
        train = pd.read_csv(TRAIN_FILE)

    if not path.exists(TEST_FILE):
        if path.exists(TEST_FILE + ".tmp"):            
            test = pd.read_csv(TEST_FILE + ".tmp")
        else:
            test = pd.read_csv(PRE_TEST_FILE, converters=convs)
            test = do_risk(test)
            # save a tmp file for safety in the case of a further error
            test.to_csv(TEST_FILE + ".tmp", sep=',',na_rep="NA")
        cols = list(test.columns.values)
        print cols
        for c in cols:
            if c.startswith('Unnamed'):
                test = test.drop(c,1)
                print c, "droped"
        #some test location NAs
        imp = Imputer(strategy='median',axis=0)
        for state in np.unique(test.state):
            v = test[test['state']==state].values
            # sklearn bug version 0.14.1 - need to stack a dummy column before median imputation
            # see http://stackoverflow.com/questions/23742005/scikit-learn-imputer-class-possible-bug-with-median-strategy
            z = np.zeros(len(v))
            z = z.reshape((len(z),1))
            v = np.hstack((z,v))
            v = imp.fit_transform(v)
            test[test['state']==state] = v
        test.to_csv(TEST_FILE, sep=',',na_rep="NA")  
    
    else:
        test = pd.read_csv(TEST_FILE)
        
    print train.shape, test.shape
    return train, test


def do_risk(dt):
    state, old_state = "FL",""
    age_youngest = 75
    age_oldest = 0
    print "You'd better off drink a beer .... it will take a while ....."
    sleep(2)
    t0 = time()
    for i in range(dt.shape[0]):
        risk_factor = dt['risk_factor'][i]
        if np.isnan(risk_factor):
            state, age_oldest, age_youngest = dt['state'][i], dt['age_oldest'][i],dt['age_youngest'][i]
            if state <> old_state:
                q_state = dt[(dt['state']==state) & (~np.isnan(dt['risk_factor']))]
            old_state = state

            q = q_state[((q_state['age_youngest']==age_youngest) & (q_state['age_oldest']==age_oldest))]
            if len(q) > 0:
                v = q['risk_factor'].median()
                if np.isnan(v):
                    print i,"ISNAN"
                    print q
                dt['risk_factor'][i] = v
            else:
                for l,off in enumerate([1,2,3,4]):                
                    q = q_state[((q_state['age_youngest']>=(age_youngest - off)) & (q_state['age_youngest'] <=(age_youngest + off)))\
                                & ((q_state['age_oldest']>=(age_oldest - off)) & (q_state['age_oldest']<=(age_oldest + off)))]
                    if len(q) > 0:
                        dt['risk_factor'][i] = q['risk_factor'].median()
                        print i,":::LEVEL %i::::" % (l+1), len(q_state), len(q), state, age_youngest, age_oldest
                        break
                if len(q) == 0:
                    q = dt[((dt['age_youngest']>=(age_youngest - off)) & (dt['age_youngest'] <=(age_youngest + off)))\
                        & ((dt['age_oldest']>=(age_oldest - off)) & (dt['age_oldest']<=(age_oldest + off)))]
                    if len(q) > 0:
                        dt['risk_factor'][i] = q['risk_factor'].median()
                        print i,":::LEVEL %i::::" % (l+2), len(q_state), len(q), state, age_youngest, age_oldest
                    else:
                        if len(q) > 0:
                            q = dt[((dt['age_youngest']==age_youngest) & (dt['age_oldest']==age_oldest) & (~np.isnan(dt['risk_factor'])))]
                            print i,":::LEVEL %i::::" % (l+3), len(q_state), len(q), state, age_youngest, age_oldest
                        else:
                            print i,":::FAILED::::", len(q_state), len(q), state, age_youngest, age_oldest
                            break

    print "risk NA done in %2.2f s" % (time() - t0)
    print dt.shape
    return dt

def main():
    print __doc__
    pre_parse()    
    
if __name__ == '__main__':
    main()



