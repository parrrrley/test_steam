
# 필요 패키지 추가
import time
import datetime
import pickle
import glob

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.font_manager as fm
import copy
import my_func as my

#한글깨짐 방지코드 
font_location = '/home/sagemaker-user/gsc/NanumGothic.ttf'
fm.fontManager.addfont(font_location)
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)
matplotlib.rc('axes', unicode_minus=False)


# 웹 페이지 기본 설정
# page title: 데이터 분석 및 모델링 대시보드
st.set_page_config(
    page_title="Steam Imbalance Mornitoring", # page 타이틀
    page_icon="🧊", # page 아이콘
    layout="wide", # wide, centered
    initial_sidebar_state="auto", # 사이드 바 초기 상태
    menu_items={
        'Get Help': 'http://s2cimp.gscaltex.co.kr/monitoring/energy/SteamBalance_D',
        'Report a bug': None,
        'About': '2023 GS CDS Class',

    }
)

# 실습 소개 페이지 출력 함수
# 소개 페이지는 기본으로 제공됩니다.
def front_page():
    st.title('Steam Imbalance Mornitoring')
    st.write('이 페이지는 스팀 밸런스 데이터 분석, 학습 및 서빙 대시보드를 생성합니다.')
    st.markdown(' 1. EDA 페이지 생성')
    st.markdown('''
        - 데이터 로드 (파일 업로드)
        - 데이터 분포 확인 (시각화)
        - 데이터 관계 확인 (개별 선택, 시각화)
    ''')
    st.markdown(' 2. Modeling 페이지 생성')
    st.markdown('''
        - 변수 선택 및 데이터 분할
        - 모델링 (하이퍼 파라미터 설정)
        - 모델링 결과 확인 (평가 측도, 특성 중요도)
    ''')
    st.markdown(' 3. Model Serving 페이지 생성')
    st.markdown('''
        - 입력 값 설정 (메뉴)
        - 모델 임발란스 확인
        - 임발란스 비교
    ''')    
    
# 1. file load 함수
# 2. 파일 확장자에 맞게 읽어서 df으로 리턴하는 함수
# 3. 성능 향상을 위해 캐싱 기능 이용
@st.cache_data
def load_file(file):
    
    # 확장자 분리
    ext = file.name.split('.')[-1]
    name = file.name.split('.')[0]
    # 확장자 별 로드 함수 구분
    if ext == 'csv':
        k = {name : pd.read_csv(file)}
        return k
    elif 'xls' in ext:
        return pd.read_excel(file, sheet_name=None, engine='openpyxl')

# file uploader 
# session_state에 다음과 같은 3개 값을 저장하여 관리함
# 1. st.session_state['eda_state'] = {}
#  1.1 : st.session_state['eda_state']['current_file']  / st.session_state['eda_state']['current_data']
# 2. st.session_state['modeling_state'] = {}
# 3. st.session_state['serving_state'] = {}
def file_uploader():
    # 파일 업로더 위젯 추가
    file = st.file_uploader("파일 선택(csv or excel)", type=['csv', 'xls', 'xlsx'])
    
    if file is not None:
        # 새 파일이 업로드되면 기존 상태 초기화
        st.session_state['eda_state'] = {}
        st.session_state['modeling_state'] = {}
        st.session_state['serving_state'] = {}
        
        # 새로 업로드된 파일 저장
        st.session_state['eda_state']['current_file'] = file
    
    # 새로 업로드한 파일을 dict:df로 로드
    if 'current_file' in st.session_state['eda_state']:
        st.write(f"Current File: {st.session_state['eda_state']['current_file'].name}")
        st.session_state['eda_state']['current_data_dict'] = load_file(st.session_state['eda_state']['current_file'])
     
 
    # 새로 로드한 df_raw 저장
    if 'current_data_dict' in st.session_state['eda_state']:
        sheet = st.selectbox("분석할 Sheet 선택", st.session_state['eda_state']['current_data_dict'].keys())
        st.session_state['eda_state']['current_sheet'] = sheet
        st.session_state['eda_state']['current_data_raw'] = st.session_state['eda_state']['current_data_dict'][sheet]
        # 새로 로드한 df 저장
        raw = st.session_state['eda_state']['current_data_raw']
        st.session_state['eda_state']['current_data'] = my.rawDf(raw)

        # 생산/고정 변수 default 목록 저장
        st.session_state['eda_state']['생산'], st.session_state['eda_state']['고정_default'] = my.defCol(raw)
        # 화면에 dataFrame 표기
        st.dataframe(st.session_state['eda_state']['current_data'])
        
    # 전체/헤더별 변수 Dictionary 생성
    if 'current_data_dict' in st.session_state['eda_state']:
        st.session_state['eda_state']['coef_whole'] = dict()
        st.session_state['eda_state']['coef_header'] = dict()
                    

def select_variables():
    if 'current_data' in st.session_state['eda_state']:
        df = st.session_state['eda_state']['current_data']
        cols = df.columns[1:]
        고정=st.session_state['eda_state']['고정_default']
        coef_whole = st.session_state['eda_state']['coef_whole']
        
        # 이전 단계 헤더에서 생성한 coef 승계하여 df 조정 
        df, 고정, 승계 = my.getCol(df, 고정, coef_whole)
        st.write(f'이전 단계 고정 변수(고정 유지 추천): {승계}')
        
        # 고정할 변수(y값) 선택             
        options = st.multiselect(
        '고정할 변수 선택 : 생산량(기준값 y)로 병합',
        cols,
        default=고정,
        key="unique_multiselect_key_select_varaiables",
        max_selections=len(cols))

        # 고정된 변수들 merge하여 y값 생성, 고정 변수 조정할 df 저장

        생산 = st.session_state['eda_state']['생산']
        st.session_state['eda_state']['current_data_fixed'] = my.mergeCol(df, options, 생산)
        st.session_state['modeling_state']['selected_features'] = st.session_state['eda_state']['current_data_fixed'].columns[1:-1]
        st.session_state['modeling_state']['selected_labels'] = st.session_state['eda_state']['current_data_fixed'].columns[-1]

        sheet = st.session_state['eda_state']['current_sheet']
        selected_features = list(st.session_state['modeling_state']['selected_features'])
        st.write(f"선택된 독립 변수: {selected_features}")
        st.write(f'Current Sheet: {sheet}')
        st.dataframe(st.session_state['eda_state']['current_data_fixed'])
    else:
        st.write('먼저 파일을 업로드해주세요')

# get_info 함수
@st.cache_data
def get_info(col, df):
    # 독립 변수 1개의 정보와 분포 figure 생성 함수
    plt.figure(figsize=(1.5,1))
    
    # 전부 수치형 변수(int64, float64)는 histogram : sns.histplot() 이용
    ax = sns.histplot(x=df[col], bins=20)
    plt.grid(False)
    # 범주형 변수는 seaborn.barplot 이용

    plt.xlabel('')
    # plt.xticks([])
    plt.ylabel('count')
    sns.despine(bottom = True, left = True)
    fig = ax.get_figure()
    
    # 사전으로 묶어서 반환
    return {'name': col, 'total': df[col].shape[0], 'na': df[col].isna().sum(), 'type': df[col].dtype, 'distribution':fig }
        
# variables 함수
def variables():
    # 각 변수 별 정보와 분포 figure를 출력하는 함수
    
    # 저장된 df가 있는 경우에만 동작
    if 'current_data' in st.session_state['eda_state']:
        df = st.session_state['eda_state']['current_data']
        cols = df.columns[1:]

        # 열 정보를 처음 저장하는 경우 초기 사전 생성
        if 'column_dict' not in st.session_state['eda_state']:
            st.session_state['eda_state']['column_dict'] = {}
        
        # EDA 확인할 변수 설정
        options = st.multiselect('변수 선택',cols,[], max_selections=len(cols))
        
        # 선택된 열에 대한 정보 생성 후 저장
        for col in options:
            st.session_state['eda_state']['column_dict'][col] = get_info(col, df)

        # 각 열의 정보를 하나씩 출력
        for col in st.session_state['eda_state']['column_dict']:
            with st.expander(col, expanded=True):
                left, center, right = st.columns((1, 1, 1.5))
                right.pyplot(st.session_state['eda_state']['column_dict'][col]['distribution'], use_container_width=True)
                left.subheader(f"**:blue[{st.session_state['eda_state']['column_dict'][col]['name']}]**")
                left.caption(st.session_state['eda_state']['column_dict'][col]['type'])
                cl, cr = center.columns(2)
                cl.markdown('**Missing**')
                cr.write(f"{st.session_state['eda_state']['column_dict'][col]['na']}")
                cl.markdown('**Missing Rate**')
                cr.write(f"{st.session_state['eda_state']['column_dict'][col]['na']/len(df):.2%}")
    else:
        st.write('먼저 파일을 업로드해주세요')

# corr 계산 함수
@st.cache_data

def get_corr(options, df):
    # 전달된 열에 대한 pairplot figure 생성
    corr_matrix = df[options].corr()
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(corr_matrix, cmap='RdBu_r', vmin=-1, cbar_kws={'shrink': 0.7}, annot=True, fmt='.2f', ax=ax)
    return fig
            
# correlation tab 출력 함수
def correlation():
    cols = []
    
    # 저장된 df가 있는 경우에만 동작
    if 'current_data_fixed' in st.session_state['eda_state']:
        df = st.session_state['eda_state']['current_data_fixed']
        cols = df.columns[1:]
        # 상관 관계 시각화를 할 변수 선택 (2개 이상)
        options = st.multiselect(
        '변수 선택',
        cols,
        [],
        key="unique_multiselect_key",
        max_selections=len(cols))
        # 선택된 변수가 2개 이상인 경우 figure를 생성하여 출력
        if len(options)>=2:
            st.pyplot(get_corr(options, df))

    else:
        st.write('먼저 Select Variables에서 고정할 변수를 지정해주세요')
  

        
def missing_data():
    pass
            
# EDA 페이지 출력 함수
def eda_page():
    st.title('Exploratory Data Analysis')
    
    # eda page tab 설정
    # tabs에는 File Upload, Variables (type, na, 분포 등), Correlation(수치)이 포함됩니다.
    t1, t2, t3, t4 = st.tabs(['File Upload', 'Select Variables', 'Variables','Correlation'])
    
    with t1:
        file_uploader()
    
    with t2:
        select_variables()
    
    with t3:
        variables()
        
    with t4:
        correlation()
        
        
# 독립, 고정 변수 선택 및 데이터 분할 함
def select_split():
    cols = []
    selected_features = []
    selected_label = []
    split_rate = 0
    
    # 저장된 df가 있는 경우에만 실행
    if 'current_data_fixed' in st.session_state['eda_state']:
        df = st.session_state['eda_state']['current_data_fixed']
        cols = df.columns
    # 이미 저장된 선택된 독립 변수 그대로 출력        
        selected_features = list(st.session_state['modeling_state']['selected_features'])
        selected_label = st.session_state['modeling_state']['selected_labels']
        st.write(f'선택된 독립 변수: {selected_features}')
        st.write(f'선택된 종속 변수: {selected_label}')
        
        # 학습/제외할 데이터 기간 선택 
        df1 = df.copy()
        selected_dates = st.date_input(
        '학습할 기간 선택',
        min_value=df.iloc[0,0],
        max_value=df.iloc[-1,0],
        value=(df.iloc[0,0],df.iloc[-1,0]),
        key='selected_dates')
        
        if selected_dates:
            df2 = my.defTime(df1, selected_dates[0], selected_dates[1])
            excepted_dates = st.date_input(
            '제외할 기간 선택1',
            min_value=df2.iloc[0,0],
            max_value=df2.iloc[-1,0],
            value=(df2.iloc[0,0],df2.iloc[0,0]),
            key='excepted_dates1',
            )

            if excepted_dates:
                df2 = my.removeTime(df2, excepted_dates[0], excepted_dates[1])
                excepted_dates2 = st.date_input(
                '제외할 기간 선택2',
                min_value=df2.iloc[0,0],
                max_value=df2.iloc[-1,0],
                # value=None,
                key='excepted_dates2', value=(df2.iloc[0,0],df2.iloc[0,0])
                )
                if excepted_dates2:
                    df2 = my.removeTime(df2, excepted_dates2[0], excepted_dates2[1])

                    excepted_dates3 = st.date_input(
                    '제외할 기간 선택3',
                    min_value=df2.iloc[0,0],
                    max_value=df2.iloc[-1,0],
                    # value=None,
                    key='excepted_dates3', value=(df2.iloc[0,0],df2.iloc[0,0])
                    )

                    df2 = my.removeTime(df2, excepted_dates3[0], excepted_dates3[1])
        st.session_state['eda_state']['current_data_fixed2'] = df2
        st.dataframe(df2)
    else:
        st.write('EDA Page의 Select Variables를 먼저 진행해주세요')
        


# 하이퍼 파라미터 설정
def set_hyperparamters(model_name):
    param_list = {
        'Linear Regressor':{
            'fit_intercept':[0, 1, 0, 1],
            'epochs':[1, 1000, 100, 1]},
        'Ridge Regressor':{
            'alpha' : [0.0, 10.0, 0.1, 0.01],
            'fit_intercept':[0, 1, 0, 1],
            'epochs':[1, 1000, 100, 1]},
        'Lasso Regressor':{
            'alpha' : [0.0, 10.0, 0.1, 0.01],
            'fit_intercept':[0, 1, 0, 1],
            'epochs':[1, 1000, 100, 1]}
    }
    ret = {}
    with st.form('hyperparameters'):
        st.write('alpha: Ridge와 Lasso에서 높을수록 기울기가 0에 가깝게 규제 (0~1 범위 추천)')
        st.write('fit_intercept : 상수항(y절편) 생성 여부, 0 지정시 imbalance의 기간 평균값이 상수항')
        st.write('epochs: 학습 횟수 (100 이상 추천) 높을수록 학습 시간 소요')
        
        for key, item in param_list[model_name].items():
            ret[key] = st.slider(key, *item)
        if ret['fit_intercept'] ==1:
            ret['fit_intercept'] = True
        else : 
            ret['fit_intercept'] = False
        submitted = st.form_submit_button('Run')
        
        if submitted:
            st.write(ret)
            return ret

# train_model
def train_model(selected_model, model_name):
    with st.spinner('데이터 분할 중...'): 
        st.write(st.session_state['eda_state']['current_data_fixed2'].dtypes)
        df = my.avrIm(st.session_state['eda_state']['current_data_fixed2'])
        df.iloc[:, 1:] = df.iloc[:, 1:].astype('float')
        st.write(df.dtypes)
        st.session_state['eda_state']['current_data_fixed3'] = df
        time.sleep(1)
    st.success('분할 완료')
    time.sleep(1)

    with st.spinner('학습 중...'): 
        model, coefs = selected_model(df, **st.session_state['modeling_state']['hyperparamters'])
        model.coef_ = coefs.agg('mean')[:-2]
        model.intercept_ = coefs.agg(['mean']).iloc[0, -2]
    st.success('학습 완료')
    
    with st.spinner('보정된 값 생성 중...'):
        df1 = my.newcoef(df, coefs)
        st.session_state['modeling_state']['fixed_data'] = df1
        st.session_state['eda_state']['coef_header'] = coefs.agg(['mean'])
    st.success('보정된 값 생성 완료')
    # 생성된 계수 표기
    st.markdown('학습된 계수 및 상수입니다')
    st.dataframe(coefs.agg(['mean']))
    # 계수 분포 표기
    with st.expander('계수 분포', expanded=True):
        fig, ax = plt.subplots(figsize=(20, 15))
        plot = sns.boxplot(data=coefs.iloc[:, :-2], ax=ax)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        st.pyplot(fig, use_container_width=True)
        # plt.figure(figsize=(20,15))
        # plot = sns.boxplot(data=coefs.iloc[:,:-2])
        # plt.xticks(rotation=90)
        # plt.show()
        # fig = plot.fig
        # st.pyplot(fig, use_container_width=True)
    
    # print(f'Origin R2: {r2_score(y_real, origin_pred):.4f} New Model R2: {r2_score(y_real, model_pred):.4f}')
    # print(f'Origin MAE: {mean_absolute_error(y_real, origin_pred):.4f} New Model MAE: {mean_absolute_error(y_real, model_pred):.4f}')
    # print(f'Origin MAPE: {mean_absolute_percentage_error(y_real, origin_pred):.4f} New Model MAE: {mean_absolute_percentage_error(y_real, model_pred):.4f}')
    # print(f'Origin RMSE: {mean_squared_error(y_real, origin_pred, squared=False):.4f} New Model MAE: {mean_squared_error(y_real, model_pred, squared=False):.4f}')
    file_name = f"{st.session_state['eda_state']['current_sheet']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    
#     # 모델 파일 저장
#     with open(f'./models/model_{model_name.replace(" ", "_")}_{file_name}.dat', 'wb') as f:
#         pickle.dump(model, f)
    
#     # 학습에 사용된 독립 변수 계수 저장 (순서)
#     with open(f'./models/meta_{model_name.replace(" ", "_")}_{file_name}.dat', 'wb') as f:
#         pickle.dump(coefs.agg['mean'], f)
        
    return model, coefs.agg(['mean'])
        
# modeling 함수
def modeling():
    # 모델링 tab 출력 함수
    model_list = ['Select Model', 'Linear Regressor', 'Ridge Regressor', 'Lasso Regressor']
    model_dict = {'Linear Regressor': my.learnLR, 'Ridge Regressor': my.learnRidge, 'Lasso Regressor': my.learnLasso}
    selected_model = ''
    
    if 'selected_model' in st.session_state['modeling_state']:
        selected_model = st.session_state['modeling_state']['selected_model']
    if 'hyperparamters' in st.session_state['modeling_state']:
        hps = st.session_state['modeling_state']['hyperparamters']
    if 'current_data_fixed2' in st.session_state['eda_state']:
        selected_model = st.selectbox(
        '학습에 사용할 모델을 선택하세요.',
        model_list, 
        index=0)
    else:
        st.write('먼저 Data Selection and Split을 진행해주세요')

    if selected_model in model_list[1:]:
        
        st.session_state['modeling_state']['selected_model'] = selected_model
        hps = set_hyperparamters(selected_model)
        st.session_state['modeling_state']['hyperparamters'] = hps
        
        if hps != None:
            model,coefs = train_model(model_dict[selected_model], selected_model)
            sheet = st.session_state['eda_state']['current_sheet']
            st.session_state['modeling_state']['model'] = {}
            st.session_state['modeling_state']['model'][sheet] = model
            
            st.success('학습 종료')

# 결과 tab 함수
def results():
    with st.expander('Metrics', expanded=True):
        sheet = st.session_state['eda_state']['current_sheet']
        if 'current_data_fixed3' in st.session_state['eda_state']:
            df = st.session_state['eda_state']['current_data_fixed3']
            model= st.session_state['modeling_state']['model'][sheet]
            
            # 기울기가 모두 1 인 기존 모델 지정
            model2 = copy.deepcopy(model)
            for i in range(len(model.coef_)):
                model2.coef_[i] = 1
            y_real = df.iloc[:, -3]
            origin_pred = model2.predict(df.iloc[:,1:-3])
            model_pred = model.predict(df.iloc[:,1:-3])
            # 생성 모델 점수
            st.divider()
            st.caption('New Model Results')
            c1, c2, c3, c4 = st.columns(4)
            left, right = c1.columns(2)
            r2 = r2_score(y_real, model_pred)
            left.write('**:blue[$R^2$]**')
            right.write(f'{r2: 10.5f}')

            left, right = c2.columns(2)
            mae = mean_absolute_error(y_real, model_pred)
            left.write('**:blue[MAE]**')
            right.write(f'{mae: 10.5f}')

            left, right = c3.columns(2)
            mape = mean_absolute_percentage_error(y_real, model_pred)
            left.write('**:blue[MAPE]**')
            right.write(f'{mape: 10.5f}')
            
            left, right = c4.columns(2)
            rmse = mean_squared_error(y_real, model_pred, squared=False)
            left.write('**:blue[RMSE]**')
            right.write(f'{rmse: 10.5f}')
            
            # 기존 모델 점수
            st.divider()
            st.caption('Original Model(Raw Data) Results')
            c1, c2, c3, c4 = st.columns(4)
            left, right = c1.columns(2)
            r2 = r2_score(y_real, origin_pred)
            left.write('**:blue[$R^2$]**')
            right.write(f'{r2: 10.5f}')

            left, right = c2.columns(2)
            mae = mean_absolute_error(y_real, origin_pred)
            left.write('**:blue[MAE]**')
            right.write(f'{mae: 10.5f}')

            left, right = c3.columns(2)
            mape = mean_absolute_percentage_error(y_real, origin_pred)
            left.write('**:blue[MAPE]**')
            right.write(f'{mape: 10.5f}')
            
            left, right = c4.columns(2)
            rmse = mean_squared_error(y_real, origin_pred, squared=False)
            left.write('**:blue[RMSE]**')
            right.write(f'{rmse: 10.5f}')
        
            st.divider()

    with st.expander('Result Analysis', expanded=False):
        if 'current_data_fixed3' in st.session_state['eda_state']:
            df1 = st.session_state['eda_state']['current_data_fixed3'].iloc[:, 1:]
            df = df1.copy()
            df.astype('float')
            model= st.session_state['modeling_state']['model'][sheet]
            
            # 기울기가 모두 1 인 기존 모델 지정
            model2 = copy.deepcopy(model)
            for i in range(len(model.coef_)):
                model2.coef_[i] = 1
            y_real = df.iloc[:, -3]
            origin_pred = model2.predict(df.iloc[:,:-3])
            model_pred = model.predict(df.iloc[:,:-3])
            
            data1 = {
                'real': y_real,
                'prediction': model_pred
            }
            result1 = pd.DataFrame(data1)
            plot = sns.lmplot(x='real', y='prediction', data=result1, line_kws={'color':'red'})
            plt.title('New Model Results')
            fig = plot.fig
            st.pyplot(fig, use_container_width=True)

            plt.figure()

            data2 = {
                'real': y_real,
                'prediction': origin_pred
            }
            result2 = pd.DataFrame(data2)
            plot = sns.lmplot(x='real', y='prediction', data=result2, line_kws={'color':'red'})
            plt.title('Origin Model(Raw Data) Results')
            fig = plot.fig
            st.pyplot(fig, use_container_width=True)

    st.divider()
    


# Modeling 페이지 출력 함수
def modeling_page():
    st.title('ML Modeling')
    
    # tabs를 추가하세요.
    # tabs에는 File Upload, Variables (type, na, 분포 등), Correlation(수치)이 포함됩니다.
    t1, t2, t3, t4 = st.tabs(['Data Selection and Split', 'Modeling', 'Results', 'Saving'])

    # file upload tab 구현
    with t1:
        select_split()
    
    with t2:
        modeling()
    
    with t3:
        results()
        
    with t4:
        saving()

# 추론 함수
def inference():
    model = st.session_state['serving_state']['model']
    model_name = st.session_state['serving_state']['model_name']
    model_name = model_name.removeprefix('model_').replace('_', ' ')
    
    if 'meta' in st.session_state['serving_state']:
        placeholder = ', '.join(st.session_state['serving_state']['meta'])
    else:
        placeholder = ''
    
    with st.expander('Inference', expanded=True):
        st.caption(model_name)
        input_data = st.text_input(
        label='예측하려는 값을 입력하세요.',
        placeholder=placeholder)

        if input_data:
            input_data = [[float(s) for s in input_data.split(',')]]

            left, center, right = st.columns(3)
            left.write('**:blue[입력]**')
            center.write(input_data)

            left, center, right = st.columns(3)
            left.write('**:blue[출력]**')
            center.write(model.predict(np.array(input_data)))

# Serving 페이지 출력 함수
def serving_page():
    st.title('ML Serving')
    
    with st.form('select pre-trained model'):
        # 모델 파일 목록 불러오기
        model_paths = glob.glob('./models/model_*')
        model_paths.sort(reverse=True)
        model_list = [s.removeprefix('./models/').removesuffix('.dat') for s in model_paths]
        model_dict = {k:v for k, v in zip(model_list, model_paths)}
        model_list = ['Select Model'] + model_list
        
        # 추론에 사용할 모델 선택
        
        selected_inference_model = st.selectbox('추론에 사용할 모델을 선택하세요.', model_list, index=0)
        checked = st.checkbox('독립 변수 정보')
        
        submitted = st.form_submit_button('Confirm')
        
        if submitted:
            st.session_state['serving_state'] = {}
            with open(model_dict[selected_inference_model], 'rb') as f_model:
                inference_model = pickle.load(f_model)
                st.session_state['serving_state']['model'] = inference_model
                st.session_state['serving_state']['model_name'] = selected_inference_model
                if checked:
                    with open(model_dict[selected_inference_model].replace('model_', 'meta_'), 'rb') as f_meta:
                        metadata = pickle.load(f_meta)
                        st.session_state['serving_state']['meta'] = metadata
                placeholder = st.empty()
                placeholder.success('모델 불러오기 성공')
                time.sleep(2)
                placeholder.empty()
                
    if 'model' in st.session_state['serving_state']:
        inference()
        
                
# session_state에 사전 sidebar_state, eda_state, modeling_state, serving_state를 추가하세요.
if 'sidebar_state' not in st.session_state:
    st.session_state['sidebar_state'] = {}
    st.session_state['sidebar_state']['current_page'] = front_page
if 'eda_state' not in st.session_state:
    st.session_state['eda_state'] = {}
if 'modeling_state' not in st.session_state:
    st.session_state['modeling_state'] = {}
if 'serving_state' not in st.session_state:
    st.session_state['serving_state'] = {}
    
# sidebar 추가
with st.sidebar:
    st.subheader('Dashboard Menu')
    b1 = st.button('Front Page', use_container_width=True)
    b2 = st.button('EDA Page', use_container_width=True)
    b3 = st.button('Modeling Page', use_container_width=True)
    b4 = st.button('Serving Page', use_container_width=True)
    
if b1:
    st.session_state['sidebar_state']['current_page'] = front_page
#     st.session_state['sidebar_state']['current_page']()
    front_page()
elif b2:
    st.session_state['sidebar_state']['current_page'] = eda_page
#     st.session_state['sidebar_state']['current_page']()
    eda_page()
elif b3:
    st.session_state['sidebar_state']['current_page'] = modeling_page
#     st.session_state['sidebar_state']['current_page']()
    modeling_page()
elif b4:
    st.session_state['sidebar_state']['current_page'] = serving_page
#     st.session_state['sidebar_state']['current_page']()
    serving_page()
else:
    st.session_state['sidebar_state']['current_page']()
