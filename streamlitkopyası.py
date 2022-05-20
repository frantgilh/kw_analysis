import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import plotly.express as px
import xgboost as xgb
import datetime
import sys
#sys.path.append('../utils')
import kwanalysis as kw

geo = 'TR'
pytrends = TrendReq(hl=geo, tz=360,timeout=(8,15),retries = 10)

brands = ['hepsiburada ','trendyol ']
relkws = ['xiaomi','apple','samsung','huawei','iphone','ipad','telefon','bilgisayar','tablet','televizyon','laptop','notebook','monitör','bluetooth kulaklık','akıllı saat',
          'klavye','mouse',          
          'ayakkabı','saat','elbise','kadın','abiye','mont','pantolon','ceket','kadın ceket','kadın pantolon','kadın ayakkabı','kadın mont','kadın çanta','kadın terlik',
          'kadın saat','erkek saat','güneş gözlüğü','kolye','yüzük','nevresim takımı','perde','avize','tablo','yastık','ofis','ofis koltuğu','çalışma masası','kırtasiye',
          'erkek ceket','erkek pantolon','erkek ayakkabı','erkek mont','erkek terlik','yemek takımı','çaydanlık','tencere','bardak','yatak','koltuk','masa','sandalye',
          'koton','mavi','adidas','nike','lcw','erkek çocuk','kız çocuk','ütü','elektrik süpürgesi','tost makinesi','kahve makinesi','blender',
          'indirim','efsane cuma',]
pairs= []

for i in relkws:
    temp = []
    for j in brands:
        temp.append(j+i)
    pairs.append(temp)
    
    
unst = [i for j in pairs for i in j]



def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    #df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','quarter','month','year',#,'dayofweek'
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X



# --------------------------------- HEATMAP --------------------------------------------
placeholder = st.empty()
if st.sidebar.button('Heatmap'):
    placeholder.empty()
    df = pd.read_excel('heatmap.xlsx',index_col=0)
    from  matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
    pd.set_option('display.precision', 2)
    placeholder.dataframe(df.style.background_gradient(cmap=cmap),1000,500)
# --------------------------------------------------------------------------------------

# HAZIRDAN OKUMA ALANI

# --------------------------------- unst KW Seçimi -------------------------------------
# Forecast için tarih deneyebilirsin
st.sidebar.header('HAZIR KW LER')
kwsec1 = st.sidebar.expander('KW Seçimi', expanded=False)

keyword = kwsec1.selectbox(
     'Kw Seçiniz',
     (unst))
data = kw.gtrends_([keyword])

decomp, forecast = kwsec1.columns(2)
split_date = kwsec1.date_input(
     "split date",
     datetime.date(2021, 6, 1))
if decomp.button('Decomposition'):
    data = kw.gtrends_([keyword])
    placeholder.empty()
    placeholder.plotly_chart(kw.plot_decomposition_px(data,keyword,isreturn=True))
if forecast.button('Forecast'):
    placeholder.empty()
    split_date = pd.to_datetime(split_date)
    a_train = data.loc[data.index <= split_date].copy()
    a_test = data.loc[data.index > split_date].copy()
    X_train, y_train = create_features(a_train, label=keyword)
    X_test, y_test = create_features(a_test, label=keyword)
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[ (X_test, y_test)],
            early_stopping_rounds=50,
        verbose=False)
    #plot_importance(reg, height=0.9)
    a_test['Prediction'] = reg.predict(X_test)
    a_all = pd.concat([a_test, a_train], sort=False)
    placeholder.plotly_chart(px.line(a_all,x='date',y=[keyword,'Prediction',]))
# --------------------------------------------------------------------------------------

# --------------------------------- ikili KW Seçimi -------------------------------------

kwsec2 = st.sidebar.expander('KW Seçimi', expanded=False)

keyword = kwsec2.selectbox(
     'Kw Seçiniz',
     (pairs))
print(keyword)
data = kw.gtrends_(keyword)
data['pair'] = data[keyword[0]]/data[keyword[1]]
data['pair'].replace(np.inf, 0,inplace=True)

decomp2, forecast2 = kwsec2.columns(2)
if decomp2.button('Decomposition2'):
    placeholder.empty()
    placeholder.plotly_chart(kw.plot_decomposition_px(data,'pair',isreturn=True))
if forecast2.button('Forecast2'):
    placeholder.empty()
    split_date = '2021-06-01'
    a_train = data.loc[data.index <= split_date].copy()
    a_test = data.loc[data.index > split_date].copy()
    X_train, y_train = create_features(a_train, label='pair')
    X_test, y_test = create_features(a_test, label='pair')
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[ (X_test, y_test)],
            early_stopping_rounds=50,
        verbose=False)
    #plot_importance(reg, height=0.9)
    a_test['Prediction'] = reg.predict(X_test)
    a_all = pd.concat([a_test, a_train], sort=False)
    placeholder.plotly_chart(px.line(a_all,x='date',y=['pair','Prediction',]))
    


# SORGU ATIP ÇEKME ALANI

# --------------------------------- unst KW Seçimi -------------------------------------
# Forecast için tarih deneyebilirsin
st.sidebar.header('KW ARAMA')
kwsec3 = st.sidebar.expander('KW Seçimi', expanded=False)

keyword = kwsec3.text_input('KW yazınız')



decomp3, forecast3 = kwsec3.columns(2)
if decomp3.button('Decomposition3'):
    data = kw.gtrends_([keyword])
    placeholder.empty()
    placeholder.plotly_chart(kw.plot_decomposition_px(data,keyword,isreturn=True))
if forecast3.button('Forecast3'):
    data = kw.gtrends_([keyword])
    placeholder.empty()
    split_date = '2021-06-01'
    a_train = data.loc[data.index <= split_date].copy()
    a_test = data.loc[data.index > split_date].copy()
    X_train, y_train = create_features(a_train, label=keyword)
    X_test, y_test = create_features(a_test, label=keyword)
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[ (X_test, y_test)],
            early_stopping_rounds=50,
        verbose=False)
    #plot_importance(reg, height=0.9)
    a_test['Prediction'] = reg.predict(X_test)
    a_all = pd.concat([a_test, a_train], sort=False)
    placeholder.plotly_chart(px.line(a_all,x='date',y=[keyword,'Prediction',]))
# --------------------------------------------------------------------------------------

# --------------------------------- ikili KW Seçimi -------------------------------------

kwsec4 = st.sidebar.expander('KW Seçimi', expanded=False)

keyword = kwsec4.text_input('KW ikilisi yazınız')

keyword = keyword.split(',')
print(keyword)
decomp4, forecast4 = kwsec4.columns(2)
if decomp4.button('Decomposition4'):
    data = kw.gtrends_(keyword)
    data['pair'] = data[keyword[0]]/data[keyword[1]]
    data['pair'].replace(np.inf, 0,inplace=True)
    data.fillna(0,inplace = True)
    placeholder.empty()
    placeholder.plotly_chart(kw.plot_decomposition_px(data,'pair',isreturn=True))
if forecast4.button('Forecast4'):
    data = kw.gtrends_(keyword)
    data['pair'] = data[keyword[0]]/data[keyword[1]]
    data['pair'].replace(np.inf, 0,inplace=True)
    data.fillna(0,inplace = True)
    placeholder.empty()
    split_date = '2021-06-01'
    a_train = data.loc[data.index <= split_date].copy()
    a_test = data.loc[data.index > split_date].copy()
    X_train, y_train = create_features(a_train, label='pair')
    X_test, y_test = create_features(a_test, label='pair')
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[ (X_test, y_test)],
            early_stopping_rounds=50,
        verbose=False)
    #plot_importance(reg, height=0.9)
    a_test['Prediction'] = reg.predict(X_test)
    a_all = pd.concat([a_test, a_train], sort=False)
    placeholder.plotly_chart(px.line(a_all,x='date',y=['pair','Prediction',]))