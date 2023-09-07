import os
import string
#from turtle import width
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def generate_tables(results_path):
    save_path = f'./{results_path}/optimization/tables/'
    os.makedirs(save_path, exist_ok=True)

    excel_writer = pd.ExcelWriter(f'{save_path}tables_results.xlsx', engine='openpyxl')

    data = pd.read_csv(f'{results_path}/optimization/results.csv', index_col='Unnamed: 0')
    data = data.loc[data['imputation_method']!='gaussian_process'].reset_index(drop=True)

    # Max by imputation method
    ids_by_method = data.groupby(['imputation_method'])['auprc'].idxmax()
    data.iloc[ids_by_method].to_excel(excel_writer, sheet_name='best_by_imputation')
    excel_writer.save()

    # Max by model
    ids_by_model = data.groupby(['model'])['auprc'].idxmax()
    data.iloc[ids_by_model].to_excel(excel_writer, sheet_name='best_by_model')
    excel_writer.save()

    # Min by imputation method
    ids_by_method = data.groupby(['imputation_method'])['auprc'].idxmin()
    data.iloc[ids_by_method].to_excel(excel_writer, sheet_name='worst_by_imputation')
    excel_writer.save()

    # Min by model
    ids_by_model = data.groupby(['model'])['auprc'].idxmin()
    data.iloc[ids_by_model].to_excel(excel_writer, sheet_name='worst_by_model')
    excel_writer.save()

    excel_writer.close()

    # describe by imputation/model
    data.groupby(['imputation_method', 'norm_method', 'model'])['auprc'].describe().round(4).to_csv(f'{save_path}describe_by_imputation_norm_model.csv')

    # describe by imputation/model
    data.groupby(['model', 'norm_method', 'imputation_method'])['auprc'].describe().round(4).to_csv(f'{save_path}describe_by_model_imputation_method.csv')

def generate_graphs(results_path):
    save_path = f'{results_path}/optimization/images/'
    os.makedirs(save_path, exist_ok=True)

    data = pd.read_csv(f'{results_path}/optimization/results.csv', index_col='Unnamed: 0')
    data = data.loc[data['imputation_method']!='gaussian_process'].reset_index(drop=True)

    model_names = np.unique(data['model'])
    lower_case_remover = str.maketrans('', '', string.ascii_lowercase)
    model_ticks = {l:l.translate(lower_case_remover)for l in model_names}
    imputation_methods_ticks={
    'linear_interpolation': 'LI',
    'carry_forward': 'CF',
    'zero_imputation': 'ZI',
    'indicator_imputation': 'II',
    'forward_filling': 'FF',
}

    sns.set_theme()

    # boxplot by time and model
    plt.figure(figsize=(16,8))
    im = sns.boxplot(data=data, x = 'model', y = 'time', hue = 'model', showmeans=True, meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"}, width=0.4, dodge=False)
    locs, labels = plt.xticks()
    plt.xticks(locs, [model_ticks[label.get_text()] for label in labels], size=16)
    plt.yticks(fontsize=16)
    im.set_xlim(-1, model_names.shape[0])
    plt.legend( [],[], frameon=False)
    plt.title('Time by model', fontsize=16)
    plt.ylabel('Time (m)', fontsize=16)
    plt.xlabel('')
    im.get_figure().savefig(f'{save_path}time_by_model.png')

    # boxplot by time and model without outliers
    plt.figure(figsize=(16,8))
    im = sns.boxplot(data=data, x = 'model', y = 'time', hue = 'model', showmeans=True, meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"}, width=0.4, dodge=False, showfliers = False)
    locs, labels = plt.xticks()
    plt.xticks(locs, [model_ticks[label.get_text()] for label in labels], size=16)
    plt.yticks(fontsize=16)
    im.set_xlim(-1, model_names.shape[0])
    plt.legend( [],[], frameon=False)
    plt.title('Time by model without outliers', fontsize=16)
    plt.ylabel('Time (m)', fontsize=16)
    plt.xlabel('')
    im.get_figure().savefig(f'{save_path}time_by_model_without_outliers.png')

    # boxplot by auprc and model
    plt.figure(figsize=(16,8))
    im = sns.boxplot(data=data, x = 'model', y = 'auprc', hue = 'model', showmeans=True, meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"}, width=0.4, dodge=False)
    locs, labels = plt.xticks()
    plt.xticks(locs, [model_ticks[label.get_text()] for label in labels], size=16)
    plt.yticks(fontsize=16)
    im.set_xlim(-1, model_names.shape[0])
    plt.legend( [],[], frameon=False)
    plt.title('AUPRC by model', fontsize=16)
    plt.ylabel('AUPRC', fontsize=16)
    plt.xlabel('')
    im.get_figure().savefig(f'{save_path}auprc_by_model.png')

    # boxplot by auprc and imputation_method
    plt.figure(figsize=(16,8))
    imputations = np.unique(data['imputation_method'])
    im = sns.boxplot(data=data, x = 'imputation_method', y = 'auprc', hue = 'imputation_method', showmeans=True, meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"}, width=0.4, dodge=False, showfliers = False)
    locs, labels = plt.xticks()
    plt.xticks(locs, [imputation_methods_ticks[label.get_text()] for label in labels], rotation=0, size=16)
    plt.yticks(fontsize=16)
    im.set_xlim(-1, imputations.shape[0])
    plt.legend( [],[], frameon=False)
    plt.title('AUPRC by imputation method', fontsize=16)
    plt.ylabel('AUPRC', fontsize=16)
    plt.xlabel('')
    im.get_figure().savefig(f'{save_path}auprc_by_imputation_method.png')

    # scatter plot time, auprc and model
    order = [
        'TCN',
        'CNN',
        'MLP',
        'GRU',
        'LSTM',
        'LinearSVC',
        'XGBClassifier',
        'LogisticRegression',
        'AdaBoostClassifier',
        'RandomForestClassifier'
        ]
    d = data.groupby(['model'])[['time', 'auprc']].mean().loc[order]
    plt.figure(figsize=(16,5))
    sns.relplot(data=d, x = 'time', y = 'auprc', hue = 'model')
    plt.xlabel('Time (m)')
    plt.ylabel('AUPRC')
    plt.savefig(f'{save_path}time_auprc_by_model.png', dpi=1000)

    # bar plot with time by model and number of experiments
    count_experiments = data.groupby(['model'])['time'].count().loc[order]
    sum_times = data.groupby(['model'])['time'].sum().loc[order]
    labels = [f'{model_ticks[l]} \n({count_experiments[l]})' for l in d.index]

    plt.figure(figsize=(16,7))
    sns.barplot(model_names, sum_times, tick_label=labels)
    plt.xticks(range(len(model_names)),labels, size=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Time (m)', fontsize=16)
    plt.xlabel('')
    plt.savefig(f'{save_path}time_experiments_by_model.png')

    # bar plot with mean time by model and number of experiments
    
    sum_times = data.groupby(['model'])['time'].mean().loc[order]
    plt.figure(figsize=(16,5))
    sns.barplot(model_names, sum_times, tick_label=labels)
    locs, labels = plt.xticks()
    plt.xticks(locs, [model_ticks[label] for label in sum_times.index], size=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Time (m)', fontsize=16)
    plt.xlabel('')
    plt.savefig(f'{save_path}mean_time_experiments_by_model.png')

    # Violin plot by imputation method 1
    imputations1 = np.unique(data.imputation_method)[:3]
    data_1 = data.loc[data.imputation_method.isin(imputations1)]
    g = sns.catplot(x='model', y='auprc', hue='norm_method', data=data_1, row='imputation_method', kind='violin', aspect=6, sharey=False, height=5, legend=False, cut=0)
    g.set(ylim=(0, 1))
    g.fig.set_size_inches(25,15)
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        imputation = ax.title.get_text().split(' ')[-1]
        ax.set_title(imputation_methods_ticks[imputation], fontsize=30)
        ax.set_ylabel('AUPRC', fontsize=26) if i == 1 else ax.set_ylabel('')
        #ax.set_ylabel('')
        ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24)
    loc, labels = plt.xticks()
    plt.subplots_adjust(hspace = 0.2)
    plt.xlabel('')
    plt.xticks(loc, [model_ticks[label.get_text()] for label in labels], size=28)
    #plt.xticks(loc, ['' for _ in labels])
    plt.ylabel('')
    plt.savefig(f'{save_path}violin_plot_imputation_methods_1.png')

    # Violin plot by imputation method 2
    data_1 = data.loc[~data.imputation_method.isin(imputations1)]
    g = sns.catplot(x='model', y='auprc', hue='norm_method', data=data_1, row='imputation_method', kind='violin', aspect=6, sharey=False, height=5, legend=False, cut=0)
    g.fig.set_size_inches(25,15)
    g.set(ylim=(0, 1))
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        imputation = ax.title.get_text().split(' ')[-1]
        ax.set_title(imputation_methods_ticks[imputation], fontsize=30)
        ax.set_ylabel('AUPRC', fontsize=26) if i == 1 else ax.set_ylabel('')
        #ax.set_ylabel('')
        ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24)
    loc, labels = plt.xticks()
    plt.subplots_adjust(hspace = 0.2)
    plt.xlabel('')
    plt.xticks(loc, [model_ticks[label.get_text()] for label in labels], size=28)
    plt.ylabel('')
    plt.savefig(f'{save_path}violin_plot_imputation_methods_2.png')

def generate_graph_experiment(results_path):
    sns.set_theme()
    save_path =f'{results_path}/experiment/images/'
    data_path = f'{results_path}/experiment/'

    os.makedirs(save_path, exist_ok=True)

    x = [i for i in range(-7, 1 ,1)]

    #AUPRC table
    data = pd.read_csv(f'{data_path}auprc.csv', index_col='Unnamed: 0')
    plt.figure(figsize=(16,8))
    plt.title('Evolution of AUPRC with different horizons', size=16)
    plt.ylabel('AUPRC', size=16)
    plt.xlabel('Hours before onset', size=16)
    for model in data.index:
        sns.lineplot(x=x, y=data.loc[model].values[::-1], label=model)

    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend(fontsize=11, loc='lower right', bbox_to_anchor=(1.1, 0))
    plt.savefig(f'{save_path}auprc.png')

    #ROC_AUC table
    data = pd.read_csv(f'{data_path}roc_auc.csv', index_col='Unnamed: 0')
    plt.figure(figsize=(16,8))
    plt.title('Evolution of ROC AUC with different horizons', size=16)
    plt.ylabel('ROC AUC', size=16)
    plt.xlabel('Hours before onset', size=16)
    for model in data.index:
        sns.lineplot(x=x, y=data.loc[model].values[::-1], label=model)

    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend(fontsize=11, loc='lower right', bbox_to_anchor=(1.1, 0))
    plt.savefig(f'{save_path}roc_auc.png')

    #Accuracy table
    data = pd.read_csv(f'{data_path}accuracy.csv', index_col='Unnamed: 0')
    plt.figure(figsize=(16,8))
    plt.title('Evolution of Accuracy with different horizons', size=16)
    plt.ylabel('Accuracy', size=16)
    plt.xlabel('Hours before onset', size=16)
    for model in data.index:
        sns.lineplot(x=x, y=data.loc[model].values[::-1], label=model)

    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend(fontsize=11, loc='lower right', bbox_to_anchor=(1.1, 0))
    plt.savefig(f'{save_path}accuracy.png')



def generate_table_describe(results_path):
    save_path = f'./{results_path}/optimization/'

    data = pd.read_csv(f'{results_path}/optimization/results.csv', index_col='Unnamed: 0')
    data = data.loc[data['imputation_method']!='gaussian_process'].reset_index(drop=True)

    data.describe().to_excel(f'{save_path}describe.xlsx')
    data.describe().to_csv(f'{save_path}describe.csv')
    

    

if __name__=='__main__':
    results_path = './results/name_experiment'

    generate_tables(results_path)

    generate_graphs(results_path)

    generate_graph_experiment(results_path)