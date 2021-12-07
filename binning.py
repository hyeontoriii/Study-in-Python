# library load
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

""" 범주형 변수 EDA """
def CATEGORY_VARIABLE_EDA1(df, df_layout, variable_list):
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
    print(">>> RESULT <<<")

    for v in variable_list:

        print('Variable :', v, end='')
        fig = plt.figure(figsize=(10, 5))  ## Figure 생성 #
        ax = fig.add_subplot()
        ax2 = ax.twinx()

        pivot = df[[v, 'target']].groupby(v)
        x = pivot.mean().index
        #         y1=(pivot.count()/df[v].shape[0])['target'] # 결측치 인덱스 처리 필요
        y1 = (pivot.count() / df[v].count())['target']
        y2 = pivot.mean()['target']

        ax.set_ylim(0, 1.1)
        ax2.set_ylim(0, 1.1)

        ax.bar(x, y1, color='moccasin')
        ax2.plot(x, y2, '-s', color='red', markersize=4, linewidth=3, alpha=0.7, label='Bad(%)')

        try:
            title = df_layout.loc[df_layout['Variable'] == v]['Description_kor'].values[0]
            ax.set_title(title, fontsize=15)
        except:
            pass

        ax.set_xlabel(v, fontsize=15)
        ax.set_ylabel('Portion(%)', fontsize=13)
        ax2.set_ylabel('Bad(%)', fontsize=13)

        if len(pivot.mean().index) > 7:
            ax.tick_params(axis='x', labelsize=12, rotation=60)
        else:
            ax.tick_params(axis='x', labelsize=12)

        try:
            for i in range(len(x)):
                ax.text(x[i], y1.values[i] + 0.005, f'{y1.values[i] * 100:.1f}' + '%', fontsize=11,
                        horizontalalignment='center', color='black')
                if abs(y1.values[i] - y2.values[i]) < 0.1:
                    ax2.text(x[i], y2.values[i] + 0.1, f'{y2.values[i] * 100:.2f}' + '%', fontsize=11,
                             horizontalalignment='center', color='red', bbox={'boxstyle': 'square', 'fc': 'white'})
                else:
                    ax2.text(x[i], y2.values[i] + 0.05, f'{y2.values[i] * 100:.2f}' + '%', fontsize=11,
                             horizontalalignment='center', color='red', bbox={'boxstyle': 'square', 'fc': 'white'})

            plt.savefig('graph/category_eda_' + v + '.png')
            print('')
        except Exception as e:
            print(' >>>> ERROR')

""" 수치형 변수 EDA """
def CONTINUOUS_VARIABLE_EDA1(df, df_layout, variable_list):
    for v in variable_list:
        fig = plt.figure(figsize=(10,5))
        ax =sns.distplot(df.loc[df[v].notnull()&(df['target']==0), v], color="blue", label="Good")
        ax2=sns.distplot(df.loc[df[v].notnull()&(df['target']==1), v], color="red", label="Bad")
        ax.set_xlabel(v, fontsize=15)
        ax.set_ylabel('Density', fontsize=13)
        try:
            title = df_layout.loc[df_layout['Variable'] == v]['Description_kor'].values[0]
            ax.set_title(title, fontsize=15)
        except:
            pass

""" quantile 기반 비닝 구간 생성 함수 """
def start_bin(df, variable, q):
    # bins 없을 때
    bins = [-np.inf]
    for i in range(q, 100, q):
        b = df[variable].quantile(q=i / 100, interpolation='nearest')
        if b not in bins:
            bins.append(b)
    bins.append(np.inf)

    return bins

""" 피봇 테이블 생성 함수 """
def pivot(df, bin_list, variable):
    bins = bin_list
    bin_cut = pd.cut(df[variable], bins, right=False)
    pivot = pd.pivot_table(df, index=bin_cut, columns='target', values=variable, aggfunc='count')
    pivot['Total'] = pivot[0] + pivot[1]  # 전체건수
    pivot.columns.values[0] = "Good"  # 우량건수
    pivot.columns.values[1] = "Bad"  # 불량건수
    pivot['Prop_good'] = round(pivot['Good'] / pivot['Good'].sum() * 100, 1)  # 우량비중
    pivot['Prop_bad'] = round(pivot['Bad'] / pivot['Bad'].sum() * 100, 1)  # 불량비중
    pivot['Prop_total'] = round(pivot['Total'] / pivot['Total'].sum() * 100, 1)  # 전체비중
    pivot['Ln_odds'] = round(np.log(pivot['Good'] / pivot['Bad']), 2)  # ln(odds)
    pivot['Bad_rate'] = round(pivot['Bad'] / pivot['Total'], 2)  # 불량률
    pivot['WoE'] = round(np.log(pivot['Prop_good'] / pivot['Prop_bad']), 2)  # WoE
    pivot = pivot.reset_index(drop=False).set_index(variable, append=True)

    return pivot

""" 비닝 작업 함수 """

def merge_by_index(bin_list, start, end):
    del bin_list[start + 1:end]

    return bin_list

def merge_by_value(bin_list, value1, value2):
    start = bin_list.index(value1)
    end = bin_list.index(value2)
    del bin_list[start + 1:end]

    return bin_list

def split_by_value(bin_list, value1, value2, amount):
    start = bin_list.index(value1)

    for i in range(int((value2 - value1) / amount) - 1):
        add_value = value1 + amount * (i + 1)

        if add_value not in bin_list:
            bin_list.insert(start + 1 + i, add_value)

    return bin_list

""" BINNING 후 EDA 함수 """
def BINNING_EDA(df, df_layout, variable, bins):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()

    table = pivot(df, bins, variable)
    x=table.index.get_level_values(1).astype(str)

    y1=table['Prop_total'].values
    y2=table['Bad_rate'].values

    ax.bar(x, y1, color='moccasin')

    ax2 = ax.twinx()
    ax2.plot(x, y2, '-s', color='red', markersize=4, linewidth=3, alpha=0.7, label='Bad(%)')
    title = df_layout.loc[df_layout['Variable']==variable]['Description_kor'].values[0]
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(variable, fontsize=15)
    ax.set_ylabel('Portion(%)', fontsize=13)
    ax2.set_ylabel('Bad(%)', fontsize=13)

    ax.tick_params(axis = 'x', labelsize = 13, rotation=30)
    for i in range(len(x)):

        ax.text(x[i], y1[i], f'{y1[i]:.1f}'+'%', fontsize=11, horizontalalignment='center', color='black')
        ax2.text(x[i], y2[i], f'{y2[i]*100:.2f}'+'%', fontsize=11, horizontalalignment='center', color='red', bbox ={'boxstyle': 'square','fc': 'white'})
