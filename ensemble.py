"""
Project: http://www.kaggle.com/c/allstate-purchase-prediction-challenge
Ranking: 9th from 1571 teams
Work Period: 12-may-2014 to 19-may-2014
Author:  Euclides Fernandes Filho
email:   euclides5414@gmail.com
"""
from imports import *
from parse import parse, cols, ROOT_SUB, assertEqual

USE_CLASS = 0
USE_PROBA = 1

RE_GEN = 0 # regenerate all prediction if prediction files already exists
LOAD_POOL = 1 # multiprocess

ID_COL = 'customer_ID'

def getModel(M):
    if M.startswith('RFC'):
        return get_RFC()
    elif M.startswith('ETC'):
        return get_ETC()
    elif M.startswith('GBC'):
        return get_GBC()
    elif M.startswith('ADD'):
        return get_ADD()

def get_RFC(n_estimators=500,max_depth=None,n_jobs=-1,criterion='entropy'):
    return RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=2, min_samples_leaf=1
                ,max_features='auto', bootstrap=False, oob_score=False, n_jobs=n_jobs, min_density=None)

def get_ETC(n_estimators=500,max_depth=10,n_jobs=-1,criterion='gini'):
    return ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,min_samples_split=2, min_samples_leaf=1, max_features=None
     , bootstrap=False, oob_score=False, n_jobs=n_jobs)

def get_GBC(n_estimators=70,learning_rate=0.07,max_depth=5):
    return GradientBoostingClassifier(loss='deviance', learning_rate=learning_rate, n_estimators=n_estimators, subsample=1.0, min_samples_split=2
     , min_samples_leaf=1, max_depth=max_depth, max_features=None)

def get_ADD(n_estimators=70, learning_rate=0.05, max_depth=20, criterion='gini'):
    return AdaBoostClassifierDTC(n_estimators=n_estimators, learning_rate=learning_rate, algorithm='SAMME.R',\
            criterion=criterion, splitter='best', max_depth=max_depth, min_samples_split=2, min_samples_leaf=1, max_features=None)


def getY(X, Y_COL):
    y = X['y_%s' % Y_COL].values
    return y

def getID(X):
    return X[ID_COL]

def savefile(fname,IDs,y,mode='w'):    
    with open(ROOT_SUB + fname, mode) as f:
        if mode=='w':
            f.write('Id,Weekly_Sales\n')
        for i in range(len(IDs)):
            f.write("%s,%f\n" %(IDs[i],y[i]))
    print '\t\t\tFILE: "%s" SAVED' % (ROOT_SUB + fname) 
    return ROOT_SUB + fname

def save_predfile(fname, IDs_pred, iy_pred, iy_pred_probas, M, iy_cv=None):    
    df = pd.DataFrame(IDs_pred,columns=[ID_COL])
    if iy_cv is not None:
        df['Y_CV_%s' % Y_COL] = pd.Series(iy_cv,index=df.index)        
    df['y_%s_%s' % (Y_COL,M)] = pd.Series(iy_pred,index=df.index)
        
    for i,v in enumerate(np.unique(iy_pred)):
        df["p_%s_%i_%s" % (Y_COL,v,M)] = pd.Series(iy_pred_probas[:,i],index=df.index)
    df.to_csv(ROOT_SUB + fname ,sep=",")
    print '\t\t\tFILE: "%s" SAVED' % (ROOT_SUB + fname) 
    return df

def load_predfile(fname):
    return pd.read_csv(ROOT_SUB + fname)

def syntax(iCV,N_FOLDS,M, Y_COL):    
    return "M=%s_CV=%i-%i_COL=%s.csv" % (M,iCV,N_FOLDS,Y_COL)

def do(p):
    X_train, X_pred, ys_train, ys_pred, iCV, N_FOLDS, M, Y_COL = p
    y_col = 'y_%s' % Y_COL

    # --------------- # --------------- # ---------------             
    iy_train, iy_cv = getY(ys_train, Y_COL), (getY(ys_pred, Y_COL) if iCV else np.zeros((len(X_pred),)))
    IDs_train, IDs_pred = getID(X_train), getID(X_pred)        
    dataM = None
    # --------------- # --------------- # ---------------
    print '\t\tModel: %s' % (M)
    iX_train, iX_pred = X_train.drop(ID_COL,1), X_pred.drop(ID_COL,1)
    # --------------- # --------------- # ---------------
    fname = syntax(iCV,N_FOLDS,M, Y_COL)
    # --------------- # --------------- # ---------------
    if path.exists(ROOT_SUB + fname) and not RE_GEN:
        df = load_predfile(fname)
        iy_pred = df['y_%s_%s' % (Y_COL,M)]
        iy_cv = df['Y_CV_%s' % Y_COL]
        print '\t\t\tFILE: "%s" LOADED' % (ROOT_SUB + fname) 
    else:
        model = getModel(M)
        model.fit(iX_train,iy_train)
        iy_pred = model.predict(iX_pred)
        iy_pred_probas = model.predict_proba(iX_pred)

        y_cols = ['customer_ID']+[Y_COL]
        df = pd.DataFrame(X_pred[y_cols],columns=y_cols)
        df['Y_CV_%s' % Y_COL] = pd.Series(iy_cv,index=df.index)        
        df['y_%s_%s' % (Y_COL,M)] = pd.Series(iy_pred,index=df.index)        
        for i,v in enumerate(np.unique(iy_pred)):
            df["p_%s_%i_%s" % (Y_COL,v,M)] = pd.Series(iy_pred_probas[:,i],index=df.index)
        df.to_csv(ROOT_SUB + fname ,sep=",")
        print '\t\t\tFILE: "%s" SAVED' % (ROOT_SUB + fname) 
        
    # --------------- # --------------- # ---------------
    if iCV:
        score = F1(iy_pred,iy_cv)
        print '\t\t\tiCV:%i MODEL:%s F1:%f' % (iCV,M,score) 
    dataM = df
        # --------------- # --------------- # ---------------
    # --------------- # --------------- # ---------------

    return dataM, iCV, M
        
def stratX_CV(X, ys, N_FOLDS, Y_COL):
    
    stratifier = getY(ys,Y_COL)
    skf = StratifiedKFold(stratifier, n_folds=N_FOLDS)
    for iCV, (train_index, test_index) in enumerate(skf):
        X_train, X_cv = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        ys_train, ys_cv = ys.iloc[train_index].copy(), ys.iloc[test_index].copy()
        print '\tSPLIT iCV=%i LTrain: %i LTest: %i' % ((iCV + 1), len(train_index), len(test_index))
        yield iCV + 1, X_train, X_cv , ys_train, ys_cv


def combine(X, ys, N_FOLDS, Y_COL, MD):
    for iCV, X_train, X_cv, ys_train, ys_cv in stratX_CV(X, ys, N_FOLDS, Y_COL):
        for mMD, M in enumerate(MD):
            yield X_train, X_cv, ys_train, ys_cv, iCV, N_FOLDS, M, Y_COL

def combineM(X, X_pred, ys, N_FOLDS, Y_COL, MD):    
    for mMD, M in enumerate(MD):
        yield X, X_pred, ys, ys, 0, N_FOLDS, M, Y_COL

def work(Y_COL='G',sample_num=1,MD=None):
    n = str(sample_num)
    if MD is None: #single sample        
        MD =  ['RFC' + n,'ETC' + n,'GBC' + n,'ADD' + n]	
    N_FOLDS = 2
    t0 = time()
    print 'LOADING DATA SET ...'
    X,  X_pred, ys = parse(sample_num=sample_num)
    print 'DATA SET LOADED in %2.1f s' % (time() - t0)
    t0 = time()
    print 'LOADING/CREATING CV INDV. PREDS ...'
    data_train = {}
    params = combine(X, ys, N_FOLDS, Y_COL, MD)
    if LOAD_POOL:
        pool = Pool(processes=CORES)           
        res =  pool.map(do, params)   
        
        for dataM, iCV, M in res:
            if not data_train.has_key(iCV):
                data_train[iCV] = {}
            data_train[iCV][M] = dataM
        pool.close()
    else:
        for X_train, X_cv, ys_train, ys_cv, iCV, _ , M, Y_COL in params:
            dataM, iCV, _ = do((X_train, X_cv, ys_train, ys_cv, iCV, N_FOLDS, M, Y_COL))
            if not data_train.has_key(iCV):
                data_train[iCV] = {}
            data_train[iCV][M] = dataM     
    print 'LOADING/CREATING CV INDV. PREDS ENDED in %2.1f s' % (time() - t0)    

    t0 = time()
    print "MERGING CV INDV. PREDS ..."
    eXStack = None

    for iCV in range(1,N_FOLDS+1):
        eX = None
        for mMD, M in enumerate(MD):
            dfM = data_train[iCV][M]
            iIDs = dfM[ID_COL].values
            if eX is None:
                eX = pd.DataFrame(data=iIDs,columns=[ID_COL])
                eX['Y_CV_%s' % Y_COL] = pd.Series(dfM['Y_CV_%s' % Y_COL].values,index=eX.index)
            
            dfM = dfM.drop([ID_COL,'Y_CV_%s' % Y_COL],1)
            for c in list(dfM.columns.values):
                ill = False
                if not USE_CLASS and c.startswith('y_'):
                    dfM = dfM.drop(c,1)
                    ill = True
                if not USE_PROBA and c.startswith('p_'):
                    dfM = dfM.drop(c,1)
                    ill = True
                if c.startswith('Unnamed'):
                    dfM = dfM.drop(c,1)
                    ill = True
                if not ill:
                    eX[c] = pd.Series(data=dfM[c].values,index=eX.index)
               
            if notassertEqual(eX[ID_COL].values, iIDs, "IDs " + M): return
            
        if eXStack is None:
            eXStack = eX
        else:
            eXStack = pd.concat([eXStack,eX],axis=0)
            
    
    print 'MERGING FINAL INDV. PREDS ENDED in %2.1f s' % (time() - t0)

    iy = eXStack['Y_CV_%s' % Y_COL].values
    cols = list(eXStack.columns.values)
    cols.remove('Y_CV_%s' % Y_COL),cols.remove(ID_COL)
    for c in  [Y_COL]:
        cols.remove(c)
    print eXStack[cols + ['Y_CV_%s' % Y_COL]].head()
    iX = eXStack[cols]
    
    ensemble_model = get_ensemble_model(iX,iy,Y_COL,MD)

    print
        
    #================================ PREDICT ========================================
    t0 = time()
    print 'LOADING/CREATING FINAL INDV. PREDS ... '
    data_pred = {}
    params = combineM(X, X_pred, ys, N_FOLDS, Y_COL,MD)
    if LOAD_POOL:
        pool = Pool(processes=CORES)
        res =  pool.map(do, params)
   
        for dataM, iCV, M in res:
            data_pred[M] = dataM
        pool.close()
    else:        
        for X, X_pred, _ , _ , _, _ , M, Y_COL in params:
            dataM, iCV, M = do((X, X_pred, ys, ys, 0, N_FOLDS, M, Y_COL))
            data_pred[M] = dataM            
    print 'LOADING/CREATING FINAL INDV. PREDS ENDED in %2.1f s' % (time() - t0)

    #================================ PREDICT ========================================
    t0 = time()
    print "MERGING FINAL INDV. PREDS ..."
    eX = None

    for mMD, M in enumerate(MD):
        dfM = data_pred[M]
        iIDs = dfM[ID_COL].values
        if eX is None:
            eX = pd.DataFrame(data=iIDs,columns=[ID_COL])
            eX['Y_CV_%s' % Y_COL] = pd.Series(dfM['Y_CV_%s' % Y_COL].values,index=eX.index)
        
        dfM = dfM.drop([ID_COL,'Y_CV_%s' % Y_COL],1)
        for c in list(dfM.columns.values):
            ill = False
            if not USE_CLASS and c.startswith('y_'):
                dfM = dfM.drop(c,1)
                ill = True
            if not USE_PROBA and c.startswith('p_'):
                dfM = dfM.drop(c,1)
                ill = True
            if c.startswith('Unnamed'):
                dfM = dfM.drop(c,1)
                ill = True
            if not ill:
                eX[c] = pd.Series(data=dfM[c].values,index=eX.index)
                
        if not assertEqual(eX[ID_COL].values, iIDs, "IDs " + M): return
    print 
    print 'MERGING FINAL INDV. PREDS ENDED in %2.1f s' % (time() - t0)
    print
    Id = eX[ID_COL].values
    cols = list(eX.columns.values)
    cols.remove('Y_CV_%s' % Y_COL),cols.remove(ID_COL)
    for c in  [Y_COL]:
        cols.remove(c) 
    ieX = eX[cols]
    iXp = ieX.values
    y_pred = ensemble_model.predict(iXp)
    y_pred_probas = ensemble_model.predict_proba(iXp)

    y_cols = ['customer_ID']+['A','B','C','D','E','F','G']
    df = pd.DataFrame(X_pred[y_cols],columns=y_cols)
    df['y_%s' % Y_COL] = pd.Series(y_pred,index=df.index)
    for i,v in enumerate(np.unique(y_pred)):
        df["%s_%i" % (Y_COL,v)] = pd.Series(y_pred_probas[:,i],index=df.index)
    file_name = ROOT_SUB + "PRED_ENS_%s_%s.csv"
    df.to_csv(file_name % (Y_COL, 0), sep=",")

    return df
def get_ensemble_model(iX,iy,Y_COL,MD):        
    t0 = time()
    
    skf = StratifiedKFold(iy, n_folds=4)
    for train_index, test_index in skf:
        iX_train, iX_cv = iX.iloc[train_index], iX.iloc[test_index]
        iy_train, iy_cv = iy[train_index], iy[test_index]
        break
    print Y_COL,"ENSEMBLING ... ", iX.shape, iy.shape, iX_train.shape, iX_cv.shape
    
    model = LogisticRegression(C=0.03, fit_intercept=False, penalty='l2')
    model.fit(iX_train,iy_train)
    iy_pred = model.predict(iX_cv)
    f1 = F1(iy_cv, iy_pred)
    print "%s\tEnsemble CV:%1.2f F1: %f" % (Y_COL, (len(iy_cv)/(len(iy_train)*1.0)),f1)

    
    model = LogisticRegression(C=0.03, fit_intercept=False, penalty='l2')
    model.fit(iX,iy)
    print Y_COL,"ENSEMBLING ENDED in %2.1f s" % (time() - t0)
    
    return model


def main():
    xMD = []
    TOTAL_SAMPLES = 9
    for sample_num in range(1,TOTAL_SAMPLES+1):        
        n = str(sample_num)
        xMD = xMD + ['RFC' + n,'ETC' + n,'GBC' + n,'ADD' + n]
    # ensemble all samples
    Y_COL = 'G'
    df = work(Y_COL,1,MD=xMD)

    customer = 0
    cG = {}
    for i in range(df.shape[0]):
        y_col = "y_%s" % Y_COL
        customer = df[ID_COL][i]
        y_ipred = int(df[y_col][i])
        A,B,C,D,E,F,G = int(df['A'][i]),int(df['B'][i]),int(df['C'][i]),int(df['D'][i]),int(df['E'][i]),int(df['F'][i]),int(df['G'][i])
        
        if Y_COL == 'G':
            G = int(y_ipred)
        elif Y_COL == 'B':
            B = int(y_ipred)
        elif Y_COL == 'C':
            C = int(y_ipred)
        elif Y_COL == 'F':
            F = int(y_ipred)
        elif Y_COL == 'E':
            E = int(y_ipred)
        elif Y_COL == 'A':
            A = int(y_ipred)
        elif Y_COL == 'D':
            D = int(y_ipred)
            
        pre = "%i%i%i%i%i%i%i" % (A,B,C,D,E,F,G)
        cG[customer] = pre
        
    with open(ROOT_SUB + "%s_FINAL_PRED_ENSAMBLE.csv" % Y_COL,"w") as f:
        f.write("customer_ID,plan\n")
        for k in cG.keys():
            f.write("%i,%s\n" % (int(k),cG[k]))



if __name__ == '__main__':
    main()




