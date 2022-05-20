import pandas as pd
from pytrends.request import TrendReq
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from  matplotlib.colors import LinearSegmentedColormap


geo = 'TR'
pytrends = TrendReq(hl=geo, tz=360,timeout=(8,15),retries = 10)

"""
    !!!!! 400 HATASI VAR NEDEN BAK CHOOSE_KWS DE
"""

def gtrends(kw_list,timeframe='today 5-y',cat=0,geo='TR',pytrends=pytrends,ispair=False):
    """
    Parameters
    ----------
    kw_list : list
        Keywordlerin listesi.
    timeframe : TYPE, optional
        DESCRIPTION. The default is 'today 5-y'.
    geo : TYPE, optional
        DESCRIPTION. The default is 'TR'.
    pytrends : TYPE, optional
        DESCRIPTION. The default is 'pytrends' object.

    Returns
    -------
    data : Dataframe
        KW lere göre bireysel arama seviyeleri
    empty : list
        Çekilemeyen kw lerin listesi 
    """
    data = pd.DataFrame()
    empty = []
    for i in kw_list:
        try:
            pytrends.build_payload([i], timeframe=timeframe, geo=geo ,gprop='')
            _data = pytrends.interest_over_time()
            if _data.empty != True:
                data[i] = _data[i]
            else:
                empty.append(i)
        except :
            empty.append(i)
    return data,empty

def gtrends_(kw_list,timeframe='today 5-y',cat=0,geo='TR',pytrends=pytrends):
    pytrends.build_payload(kw_list, timeframe=timeframe, geo=geo ,gprop='')
    return pytrends.interest_over_time().drop('isPartial',axis=1)

def choose_kws(kw_list,timeframe='today 5-y',cat=0,geo='TR',pytrends=pytrends):
    last_list = []
    for kw in kw_list:
        last_list.append(choose_kw(kw))
    return last_list

def choose_kw(kw,timeframe='today 5-y',cat=0,geo='TR',pytrends=pytrends,arg_max = True):
    rel_kws = pytrends.suggestions(kw)
    _,emp = gtrends([i['mid'] for i in rel_kws],timeframe=timeframe,cat=0,geo=geo,pytrends=pytrends)
    rel = [i['mid'] for i in rel_kws]
    for x in emp:
        if x in rel:
            rel.remove(x)
    data_ = gtrends_(rel,timeframe=timeframe,cat=cat,geo=geo,pytrends=pytrends)
    for j in rel_kws:
        if j['mid'] not in emp:
            j['vol'] = int(data_[j['mid']].mean())
        else:
            j['vol'] = -1
    aa = pd.DataFrame(rel_kws)
    if arg_max:
        return aa.iloc[aa['vol'].argmax()]
    else:
        return rel_kws

def rel_query(kw,timeframe='today 5-y',geo='TR',pytrends=pytrends):
    pytrends.build_payload([kw], timeframe=timeframe, geo=geo ,gprop='')
    _data = pytrends.related_queries()
    return _data[kw]['top'],_data[kw]['rising']

def rel_topic(kw,timeframe='today 5-y',geo='TR',pytrends=pytrends):
    pytrends.build_payload([kw], timeframe=timeframe, geo=geo ,gprop='')
    _data = pytrends.related_topics()
    return _data[kw]['top'],_data[kw]['rising']

def plot1(data,plt=plt,sns=sns):
    plt.xticks(rotation=45)
    plt.title('Aranılan Hafta Oranı')
    sns.barplot(x = data.columns,y=data[data>0].count()/data.shape[0])

def plot_decomposition(data,kw,save=False):
    decomposition = seasonal_decompose(data[kw])
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(6,12))
    data[kw].plot(ax=ax1)
    ax1.set_title(f'"{kw}" Kelimesi Decomposition')
    decomposition.trend.plot(ax=ax2)
    ax2.set(ylabel='Trend',)
    decomposition.seasonal.plot(ax=ax3)
    ax3.set(ylabel='Seasonal',)
    decomposition.resid.plot(ax=ax4)
    ax4.set(ylabel='Resid',) 
    if save:
        plt.savefig(f'{kw}_decomposition.png')

def plot_forecast(data,kw,save=False):
    train = data[kw][:208]
    test = data[kw][208:]
    test = pd.DataFrame(test)
    train = pd.DataFrame(train)
    model = ARIMA(train, order=(52, 1, 0))  
    fitted = model.fit() 
    b = fitted.get_prediction(start=208,end=261).summary_frame()
    a = fitted.predict(start=1,end=261,dynamic=False)
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(a, label='forecast',color = 'r')
    plt.fill_between(b.index, b['mean_ci_lower'], b['mean_ci_upper'], color='k', alpha=0.1)
    plt.title(f'"{kw}" Kelimesi Hacim Forecasti')
    plt.xlabel("Zaman")
    plt.ylabel("Hacim")
    plt.legend()
    if save:
        plt.savefig(f'{kw}_forecast.png')

def plot_decomposition_px(data,kw,save=False,isreturn = False):
    
    decomposition = seasonal_decompose(data[kw])

    fig = make_subplots(rows=4, cols=1)
    fig.append_trace(go.Scatter(
        y=data[kw],
        x=data[kw].index,
        name='Actual'
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        y=decomposition.trend,
        x=decomposition.trend.index,
        name='Trend'
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        y=decomposition.seasonal,
        x=decomposition.seasonal.index,
        name='Seasonal'
    ), row=3, col=1)

    fig.append_trace(go.Scatter(
        y=decomposition.resid,
        x=decomposition.resid.index,
        name='Resid'
    ), row=4, col=1)

    fig.update_layout(height=1200, width=800, title_text=f'"{kw}" Kelimesi Decomposition', title_x=0.5)
    if isreturn:
        return fig
    if save:
        fig.write_html(f'{kw}_decomposition.html')
    fig.show()
    


def by_region(kw_list,timeframe='today 5-y',geo=geo,pytrends=pytrends):
    data = pd.DataFrame()
    empty=[]
    for i in kw_list:
        pytrends.build_payload([i], timeframe=timeframe,geo=geo)
        _data = pytrends.interest_by_region()
        if _data.empty != True:
            data[i] = _data[i]
        else:
            empty.append(i)
    return data[data>0].dropna(how='all').fillna(0),empty

def seasonal_heatmap(kw_list,geo=geo,timeframe='today 5-y',pytrends=pytrends,write=True):
    data, emp = gtrends(kw_list,geo=geo,timeframe=timeframe,pytrends=pytrends)
    ses_data = pd.DataFrame()
    for i in data.columns:
        ses_data[i] = seasonal_decompose(data[i]).seasonal.loc['2021-01-01':'2022-01-01']
    print('Hatalı KW ler: ' + str(emp))
    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
    pd.set_option('display.precision', 1)
    #b.style.text_gradient(cmap=cmap)
    if write:
        ses_data.reset_index(drop=True).style.background_gradient(cmap=cmap).to_excel('heatmap.xlsx')
    return ses_data.reset_index(drop=True).style.background_gradient(cmap=cmap)

def seasonal_heatmap_comp(base_brand,kw_list,geo=geo,timeframe='today 5-y',pytrends=pytrends,write=True):
    ses_data = pd.DataFrame()
    for i in kw_list:
        pytrends.build_payload([base_brand,i], timeframe=timeframe, geo=geo ,gprop='')
        data1 = pytrends.interest_over_time()
        data1 = data1.iloc[:-1,:]
        a = data1[base_brand]/(data1[i]+1)        
        ses_data[i] = seasonal_decompose(a).seasonal.loc['2021-01-01':'2022-01-01']
        cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
        pd.set_option('display.precision', 1)
    if write:
        ses_data.reset_index(drop=True).style.background_gradient(cmap=cmap).to_excel('{}_comp_heatmap.xlsx'.format(base_brand))
    ses_data.style.background_gradient(cmap=cmap)
    return ses_data

def new_seasonal_heatmap(kw_list,geo=geo,timeframe='today 5-y',pytrends=pytrends,write=True):
    data, emp = gtrends([x['mid'] for x in kw_list],geo=geo,timeframe=timeframe,pytrends=pytrends)
    ses_data = pd.DataFrame()
    for i in kw_list:
        if i['mid'] not in emp:
            ses_data[i['mid']] = seasonal_decompose(data[i['mid']]).seasonal.loc['2021-01-01':'2022-01-01']
            ses_data.rename(columns={i['mid']:i['title']},inplace=True)
    print('Hatalı KW ler: ' + str(emp))
    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)
    pd.set_option('display.precision', 1)
    if write:
        ses_data.reset_index(drop=True).style.background_gradient(cmap=cmap).to_excel('heatmap.xlsx')
    return ses_data.style.background_gradient(cmap=cmap)

def new_seasonal_heatmap_comp(base_brand,kw_list,geo=geo,timeframe='today 5-y',pytrends=pytrends,write=True):
    ses_data = pd.DataFrame()
    for i in kw_list:
        pytrends.build_payload([base_brand['mid'],i['mid']], timeframe=timeframe, geo=geo ,gprop='')
        data1 = pytrends.interest_over_time()
        data1 = data1.iloc[:-1,:]
        a = data1[base_brand['mid']]/(data1[i['mid']]+1)        
        ses_data[i['mid']] = seasonal_decompose(a).seasonal.loc['2021-01-01':'2022-01-01']
        ses_data.rename(columns={i['mid']:i['title']},inplace=True)
        cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)
        pd.set_option('display.precision', 1)
    if write:
        ses_data.reset_index(drop=True).style.background_gradient(cmap=cmap).to_excel('{}_comp_heatmap.xlsx'.format(base_brand['title']))
    return ses_data.style.background_gradient(cmap=cmap)
