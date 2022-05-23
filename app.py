import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import plotly.express as px
import xgboost as xgb
import datetime
#import sys
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
st.sidebar.header('HEATMAP')
ht_map = st.sidebar.expander("""Hafta seçerek o hafta sezonsallığı öneli olan KW leri yada KW seçerek sezonsallıkta önemli olan haftaları görebilirsiniz""", expanded=False)
placeholder = st.empty()
if ht_map.button('Heatmap'):
    placeholder.empty()
    df = pd.read_excel('heatmap.xlsx',index_col=0)
    from  matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
    pd.set_option('display.precision', 2)
    placeholder.dataframe(df.style.background_gradient(cmap=cmap),1000,500)
# --------------------------------------------------------------------------------------

df = pd.read_excel('heatmap.xlsx',index_col=0)
dic = []
for i in df.columns:
    dic.append({
        'name':i,
        'max1':df[i].nlargest(3).index[0],
        'max2':df[i].nlargest(3).index[1],
        'max3':df[i].nlargest(3).index[2],
        'min1':df[i].nsmallest(3).index[0],
        'min2':df[i].nsmallest(3).index[1],
        'min3':df[i].nsmallest(3).index[2], 
})
ddic =pd.DataFrame(dic)

week_choose = ht_map.selectbox(
    'Haftayı Seçiniz',
    set(ddic.drop('name',axis=1).values.flatten().tolist()))
if ht_map.button('Sorgula',key='week_query'):
    placeholder.empty()
    with placeholder.container():
        min1,min2,min3 = st.columns(3)
        max1,max2,max3 = st.columns(3)
    str_col = [ 
                ['min1',min1,'En Düşük 1.Sezonsallık'],
                ['min2',min2,'En Düşük 2.Sezonsallık'],
                ['min3',min3,'En Düşük 3.Sezonsallık'],
                ['max1',max1,'En Yüksek 1.Sezonsallık'],
                ['max2',max2,'En Yüksek 2.Sezonsallık'],
                ['max3',max3,'En Yüksek 3.Sezonsallık']
                ]
    for col in str_col:
        col[1].subheader(col[2])
        for i in ddic[ddic[col[0]] == week_choose].name.values:
            col[1].text(i)

kw_choose = ht_map.selectbox(
    'KW Seçiniz',
    ddic.name)
if ht_map.button('Sorgula',key='kw_query'):
    placeholder.empty()
    with placeholder.container():
        min1,min2,min3 = st.columns(3)
        max1,max2,max3 = st.columns(3)
    str_col = [ 
                ['min1',min1,'En Düşük 1.Sezonsallık'],
                ['min2',min2,'En Düşük 2.Sezonsallık'],
                ['min3',min3,'En Düşük 3.Sezonsallık'],
                ['max1',max1,'En Yüksek 1.Sezonsallık'],
                ['max2',max2,'En Yüksek 2.Sezonsallık'],
                ['max3',max3,'En Yüksek 3.Sezonsallık']
                ]
    for col in str_col:
        col[1].subheader(col[2])
        for i in ddic[ddic['name'] == kw_choose][col[0]].values:
            col[1].text(i)


# HAZIRDAN OKUMA ALANI

# --------------------------------- unst KW Seçimi -------------------------------------
# Forecast için tarih deneyebilirsin
st.sidebar.header('HAZIR KW LER')
kwsec1 = st.sidebar.expander('Seçilen KW lerin bireysel dekompozisyonunu ve tahminlemesini görebilirsiniz', expanded=False)

keyword = kwsec1.selectbox(
     'Kw Seçiniz',
     (unst))
data = kw.gtrends_([keyword])

decomp, forecast = kwsec1.columns(2)
split_date = kwsec1.date_input(
     "split date",
     datetime.date(2021, 6, 1))
if decomp.button('Decomposition',key='decomp1'):
    data = kw.gtrends_([keyword])
    placeholder.empty()
    placeholder.plotly_chart(kw.plot_decomposition_px(data,keyword,isreturn=True))
if forecast.button('Forecast',key='forecast1'):
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

kwsec2 = st.sidebar.expander('Seçilen KW lerin markalı rakip aranmalarının dekompozisyonunu ve tahminlemesini görebilirsiniz', expanded=False)

keyword = kwsec2.selectbox(
     'Kw Seçiniz',
     (pairs))
print(keyword)
data = kw.gtrends_(keyword)
data['pair'] = data[keyword[0]]/data[keyword[1]]
data['pair'].replace(np.inf, 0,inplace=True)

decomp2, forecast2 = kwsec2.columns(2)
if decomp2.button('Decomposition',key='decomp2'):
    placeholder.empty()
    placeholder.plotly_chart(kw.plot_decomposition_px(data,'pair',isreturn=True))
if forecast2.button('Forecast',key='forecast2'):
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
kwsec3 = st.sidebar.expander('Belirlenen KW ün bireysel dekompozisyonunu ve tahminlemesini görebilirsiniz', expanded=False)

keyword = kwsec3.text_input('KW yazınız')



decomp3, forecast3 = kwsec3.columns(2)
if decomp3.button('Decomposition',key='decomp3'):
    data = kw.gtrends_([keyword])
    placeholder.empty()
    placeholder.plotly_chart(kw.plot_decomposition_px(data,keyword,isreturn=True))
if forecast3.button('Forecast',key='forecast3'):
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

kwsec4 = st.sidebar.expander(' Belirlenen KW lerin markalı rakip aranmalarının dekompozisyonunu ve tahminlemesini görebilirsiniz', expanded=False)

keyword = kwsec4.text_input('KW ikilisi aralarında virgül olucak şekilde yazınız)')

keyword = keyword.split(',')
print(keyword)
decomp4, forecast4 = kwsec4.columns(2)
if decomp4.button('Decomposition',key='decomp4'):
    data = kw.gtrends_(keyword)
    data['pair'] = data[keyword[0]]/data[keyword[1]]
    data['pair'].replace(np.inf, 0,inplace=True)
    data.fillna(0,inplace = True)
    placeholder.empty()
    placeholder.plotly_chart(kw.plot_decomposition_px(data,'pair',isreturn=True))
if forecast4.button('Forecast',key='forecast4'):
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