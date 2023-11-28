import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet, LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error





def defCol(df):
    '''
    생산, 고정 변수 설정 return 생산, 고정
    return 생산,고정 List
    '''
    생산 = df.loc[4, df.iloc[2,:]=='생산'].values
    고정 = df.loc[4, (df.iloc[3,:]==1)].values

    d = pd.DataFrame(columns=고정)
    new_columns = []
    for col in d.columns:
        new_col = col
        while new_col in new_columns:  # 겹치는 경우에만 수정
            new_col = f"{new_col}.1"
        new_columns.append(new_col)
    # 고정 = d.dropna(axis=0).T
    # 고정 = np.array(고정)
    return 생산, new_columns

def rawDf(rawdf):
    '''
    rawDF에서 데이터만 가공 
    '''
    columns = rawdf.iloc[4]
    datas = rawdf.iloc[8:]
    df = pd.DataFrame(datas.values, columns=columns)

    new_columns = []
    for col in df.columns:
        new_col = col
        while new_col in new_columns:  # 겹치는 경우에만 수정
            new_col = f"{new_col}.1"
        new_columns.append(new_col)

    df.columns = new_columns

    df.dropna(axis=1, inplace=True)
    return df


def getCol(df, 고정, coef_whole):
    '''
    이전 학습결과에서 겹치는 coef 승계
    df : 정리된 데이터
    고정: 고정 변수 List
    coef_whole : 전체 누적된 Coef Dictionary
    return DataFrame, 고정, 승계된 변수
    '''
    df1 = df.copy()
    고정1 = set(고정.copy())
    coefs = coef_whole.keys()
    승계 = list()
    for col in df1.columns:
        if (col != '대기온도') and (col != '강수량'):
            if col in coefs:
                고정1.add(col)
                승계.append(col)
                df1[col] = df1[col]*coef_whole[col]
    고정2 = list(고정1)
    return df1, 고정2, 승계

# In[12]:


def mergeCol(df, 고정, 생산):
    '''
    생산 변수 +- 설정 및 고정할 변수 설정
    return DataFrame
    '''
    df.iloc[:,1:] = df.iloc[:, 1:].astype(float)
    df1 = df.copy()
    df1['생산량'] = 0
    for col in 생산:
        df1[col] = -df1[col]
    
    for col in 고정:
        df1['생산량'] =  df1['생산량'] - df1[col]
        if '대기온도' in col:
            df1['생산량'] += df1['대기온도']
            
        df1 = df1.drop(labels=[col], axis=1)
        
    return df1

# In[13]:


def defTime(df, start, end):
    '''
    start~end 시간만 남김
    return DataFream
    '''
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    df1 = df.loc[(df['Time Stamp']>=start)&(df['Time Stamp']<=end), :]
    return df1

# In[14]:


def removeTime(df, start, end):
    '''
    start~end 시간 제거
    return DataFrame
    '''
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    df1 = df.loc[(df['Time Stamp']<=start)|(df['Time Stamp']>=end), :]
    return df1

# In[15]:


def avrIm(df):
    '''
    imbalance 칼럼 생성 및 평균값 마이너스 한 df 출력
    return DataFrame
    '''
    df_copy = df.copy()  # 새로운 데이터프레임 생성
    df_copy['imbalance'] = df_copy['생산량']
    for col in df_copy.columns[1:-2]:
        if (col != '대기온도') and (col != '강수량'):
            # if pd.api.types.is_numeric_dtype(df_copy[col]):  # 열이 숫자형인지 확인
            df_copy['imbalance'] = df_copy['imbalance'] - df_copy[col]
    df_copy['im_avg'] = df_copy['imbalance'].mean()
    df_copy['생산량'] = df_copy['생산량'] - df_copy['im_avg']
    return df_copy

# In[16]:


def learnLR(df, alpha=0.1,  fit_intercept=False, epochs=100, intercept=0):
    '''
    Linear Regressor Model 학습 시작
    X에서 ['Time Stamp', '생산량', 'imbalance', 'im_avg'] 4개 제외
    return model, co(DataFrame)
    '''
    X = df.drop(['Time Stamp', '생산량', 'imbalance', 'im_avg'], axis=1)
    Y = df['생산량']
    co = pd.DataFrame()
    for i in range(1,epochs):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=i)
        model1 = LinearRegression(positive=True, fit_intercept=fit_intercept, n_jobs=-1)
        model1.fit(x_train, y_train)
        co1 = pd.Series(index=X.columns, data=model1.coef_ )
        co1 = pd.DataFrame(co1)
        co1 = co1.T
        co1['intercept'] = model1.intercept_

        co = pd.concat([co, co1], axis=0)
    co['inter'] = df['im_avg'].agg('mean')

    model1.coef_ = co.agg('mean')[:-2]
    model1.intercept_ = co.agg('mean')[-2]
    train_pred = model1.predict(x_train)
    test_pred = model1.predict(x_test)
    print(f'Train R2: {r2_score(y_train, train_pred):.4f} Test R2: {r2_score(y_test, test_pred):.4f}')
    print(f'Train MAE: {mean_absolute_error(y_train, train_pred):.4f} Test MAE: {mean_absolute_error(y_test, test_pred):.4f}')
    print(f'Train MAPE: {mean_absolute_percentage_error(y_train, train_pred):.4f} Test MAE: {mean_absolute_percentage_error(y_test, test_pred):.4f}')
    print(f'Train RMSE: {mean_squared_error(y_train, train_pred, squared=False):.4f} Test MAE: {mean_squared_error(y_test, test_pred, squared=False):.4f}')
    
    print('////   학습된 계수 및 상수입니다   ////')
    # display(co.agg(['mean']))
    return model1, co


def learnRidge(df, alpha=0.1, fit_intercept=False, epochs=100, intercept=0):
    '''
    Ridge Model 학습 시작
    X에서 ['Time Stamp', '생산량', 'imbalance', 'im_avg'] 4개 제외
    return model, co(DataFrame)
    '''
    X = df.drop(['Time Stamp', '생산량', 'imbalance', 'im_avg'], axis=1)
    Y = df['생산량']
    co = pd.DataFrame()
    for i in range(1,epochs):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=i)
        model1 = Ridge(alpha=alpha, positive=True, fit_intercept=fit_intercept, max_iter=50000)
        model1.fit(x_train, y_train)
        co1 = pd.Series(index=X.columns, data=model1.coef_ )
        co1 = pd.DataFrame(co1)
        co1 = co1.T
        co1['intercept'] = model1.intercept_

        co = pd.concat([co, co1], axis=0)
    co['inter'] = df['im_avg'].agg('mean')

    model1.coef_ = co.agg('mean')[:-2]
    model1.intercept_ = co.agg('mean')[-2]
    train_pred = model1.predict(x_train)
    test_pred = model1.predict(x_test)
    print(f'Train R2: {r2_score(y_train, train_pred):.4f} Test R2: {r2_score(y_test, test_pred):.4f}')
    print(f'Train MAE: {mean_absolute_error(y_train, train_pred):.4f} Test MAE: {mean_absolute_error(y_test, test_pred):.4f}')
    print(f'Train MAPE: {mean_absolute_percentage_error(y_train, train_pred):.4f} Test MAE: {mean_absolute_percentage_error(y_test, test_pred):.4f}')
    print(f'Train RMSE: {mean_squared_error(y_train, train_pred, squared=False):.4f} Test MAE: {mean_squared_error(y_test, test_pred, squared=False):.4f}')
    
    print('////   학습된 계수 및 상수입니다   ////')
    # display(co.agg(['mean']))
    return model1, co

def learnLasso(df, alpha=0.1, fit_intercept=False, epochs=100, intercept=0):
    '''
    Lasso Model 학습 시작
    X에서 ['Time Stamp', '생산량', 'imbalance', 'im_avg'] 4개 제외
    return model, co(DataFrame)
    '''
    X = df.drop(['Time Stamp', '생산량', 'imbalance', 'im_avg'], axis=1)
    Y = df['생산량']
    co = pd.DataFrame()
    for i in range(1,epochs):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=i)
        model1 = Lasso(alpha=alpha, positive=True, fit_intercept=fit_intercept, max_iter=50000)
        model1.fit(x_train, y_train)
        co1 = pd.Series(index=X.columns, data=model1.coef_ )
        co1 = pd.DataFrame(co1)
        co1 = co1.T
        co1['intercept'] = model1.intercept_

        co = pd.concat([co, co1], axis=0)
    co['inter'] = df['im_avg'].agg('mean')

    model1.coef_ = co.agg('mean')[:-2]
    model1.intercept_ = co.agg('mean')[-2]
    train_pred = model1.predict(x_train)
    test_pred = model1.predict(x_test)
    print(f'Train R2: {r2_score(y_train, train_pred):.4f} Test R2: {r2_score(y_test, test_pred):.4f}')
    print(f'Train MAE: {mean_absolute_error(y_train, train_pred):.4f} Test MAE: {mean_absolute_error(y_test, test_pred):.4f}')
    print(f'Train MAPE: {mean_absolute_percentage_error(y_train, train_pred):.4f} Test MAE: {mean_absolute_percentage_error(y_test, test_pred):.4f}')
    print(f'Train RMSE: {mean_squared_error(y_train, train_pred, squared=False):.4f} Test MAE: {mean_squared_error(y_test, test_pred, squared=False):.4f}')
    
    print('////   학습된 계수 및 상수입니다   ////')
    # display(co.agg(['mean']))
    return model1, co
# In[17]:


def newcoef(df, co, avrIm=True):
    '''
    학습된 coef를 데이터에 적용
    df 는 imbalance 포함 df, co는 coef + imbalance + intercept
    avrIm -> 생산량 - average Imbalance 계산 여부
    return DataFrame
    '''
    df1 = df.copy()
    for col in co.columns[:-2]:
        df1[col] = df1[col] * co.agg('mean')[col]
        
    # 생산량에 이미 average imbalance 빠져있음 
    if avrIm==True:
        df1['imbalance(보정)'] = df1['생산량']-co['intercept'].agg('mean') #-co['inter'].agg('mean')
    else:
        df1['imbalance(보정)'] = df1['생산량']-co['intercept'].agg('mean') -co['inter'].agg('mean')
        
    for col in df1.columns[1:-4]:
        df1['imbalance(보정)'] = df1['imbalance(보정)'] - df1[col]
    return df1

# In[18]:


def getScore(df):
    y_test = df['imbalance(보정)']*0
    y_train = df['imbalance']*0
    
    train_pred = df['imbalance']
    test_pred = df['imbalance(보정)'] - df['im_avg']
    print(f'Train R2: {r2_score(y_train, train_pred):.4f} Test R2: {r2_score(y_test, test_pred):.4f}')
    print(f'Train MAE: {mean_absolute_error(y_train, train_pred):.4f} Test MAE: {mean_absolute_error(y_test, test_pred):.4f}')
    print(f'Train RMSE: {mean_squared_error(y_train, train_pred, squared=False):.4f} Test MAE: {mean_squared_error(y_test, test_pred, squared=False):.4f}')

# In[19]:


def saveCoef(co, header, coef_whole, coef_header):
    '''
    co는 학습된 coef
    header는 '116LP', 'RFCCLP' 등
    coef_whole은 전체 coef dictionary
    coef_header는 헤더 별 coef dictionary 
    '''
    a = dict(co.loc[:, (co.columns!='강수량') & (co.columns!='대기온도') & (co.columns!='inter') & (co.columns!='intercept') ].agg('mean'))
    coef_whole.update(a)
    b = dict()
    b[header] = co.agg(['mean'])
    coef_header.update(b)
    print('선형회귀 계수 업데이트 완료')

# In[20]:


def loadCoef(header, coef_whole, coef_header):
    '''
    coef_header에서 헤더 별 데이터 불러와 coef_whole 에 저장    
    '''
    for h in header:
        if h in coef_header.keys():
            coef_whole.update(coef_header[h].squeeze())
            print(f'{h} 데이터 load 완료')
        else:
            print(f'{h} Header가 coef_header에 없습니다.')

# In[21]:


def loadClimate(coef_header):
    '''
    coef_header에서 intercept와 강수량/대기온도 종합
    return DataFrame
    '''
    a = pd.DataFrame(columns=['강수량', '대기온도', 'intercept', 'inter'], data=[[0,0,0,0]])
    for h in coef_header.keys():
        for col in a.columns:
            if col in coef_header[h].columns:
                a[col][0] += coef_header[h][col][0]

    return a
