import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import copy
from math import isnan


def one_hot_encode_vital_status(dataframe):
    ''' One hot encode the vital status to allow for log reg prediction.
    Alive is coded as a one, dead is coded as zero. '''
    one_hot_encoded_df = pd.get_dummies(dataframe, columns=['vital_status'],
                                        prefix='onefor')
    one_hot_encoded_df.drop(columns='onefor_dead', inplace=True)
    return one_hot_encoded_df


def split_dataframe_by_drug(drugs_of_interest, dataframe):
    ''' Splits data into a seperate drug for each drug of interest. Drugs of
    interest are a list previously determined by the user. Returns a dictionary
    with each dataframe '''
    drug_dfs = {}
    for drug in drugs_of_interest:
        drug_dfs[drug] = dataframe[dataframe.standard_drugs
                                   == drug].drop(columns='standard_drugs')
    return drug_dfs


def batch_train_test_split(drug_dfs):
    ''' For each drug dataframe, performes a train test split and returns a
    dictionary with x and y test, and x and y train '''
    split_drug_dfs = {}
    for key in drug_dfs:
        X_train, X_test, y_train, y_test = train_test_split(drug_dfs[key].drop(
                                                           'onefor_alive',
                                                            axis=1),
                                                            drug_dfs[key].
                                                            onefor_alive,
                                                            random_state=42)
        split_drug_dfs[key] = {}
        split_drug_dfs[key]['x_train'] = X_train
        split_drug_dfs[key]['x_test'] = X_test
        split_drug_dfs[key]['y_train'] = y_train
        split_drug_dfs[key]['y_test'] = y_test
    return split_drug_dfs


def run_logreg(split_drug_dfs, c):
    ''' Performs logistic regression with the given C constant, performs y
    predictions from x test, and returns a dictionary with various scores
    (score, log loss, accuracy , precision, and recall) '''
    log_loss_metric = {}
    m_score = {}
    accuracy_scores = {}
    precision_scores = {}
    recall_scores = {}
    y_preds = {}
    evaluation_metrics = {}
    m = LogisticRegression(C=c, solver='lbfgs', max_iter=1000)
    for key in split_drug_dfs:
        m.fit(split_drug_dfs[key]['x_train'], split_drug_dfs[key]['y_train'])
        mscore = m.score(split_drug_dfs[key]['x_train'],
                         split_drug_dfs[key]['y_train'])
        m_score[key] = mscore
        y_pred = m.predict(split_drug_dfs[key]['x_test'])
        y_preds[key] = y_pred
        log_loss_score = log_loss(split_drug_dfs[key]['y_test'], y_pred)
        log_loss_metric[key] = log_loss_score
        accuracy = accuracy_score(split_drug_dfs[key]['y_test'], y_pred)
        accuracy_scores[key] = accuracy
        precision = precision_score(split_drug_dfs[key]['y_test'], y_pred)
        precision_scores[key] = precision
        recall = recall_score(split_drug_dfs[key]['y_test'], y_pred)
        recall_scores[key] = recall
    evaluation_metrics['score'] = m_score
    evaluation_metrics['log_loss'] = log_loss_metric
    evaluation_metrics['accuracy'] = accuracy_scores
    evaluation_metrics['precision'] = precision_scores
    evaluation_metrics['recall'] = recall_scores
    evaluation_dataframe = pd.DataFrame(evaluation_metrics)
    return y_preds, m_score, log_loss_metric, evaluation_dataframe


def batch_min_max_scale(split_drug_dfs):
    ''' Performs min max scaling on x and y train, and x and y test. Returns
    a dictionary with the scaled dataframes '''
    # note that this cannot be used for PCA, as the mean must be centered at 0
    scaler = MinMaxScaler(feature_range=[0, 1])
    scaled_split_dfs = copy.deepcopy(split_drug_dfs)
    for key in split_drug_dfs:
        scaled_x_train = scaler.fit_transform(scaled_split_dfs[key]['x_train'])
        scaled_split_dfs[key]['x_train'] = scaled_x_train
        scaled_x_test = scaler.transform(scaled_split_dfs[key]['x_test'])
        scaled_split_dfs[key]['x_test'] = scaled_x_test
    return scaled_split_dfs


def batch_standard_scale(split_drug_dfs):
    ''' Performs standard scaling on x and y train, and x and y test. Suitable
    for use with PCA. Returns a dictionary with the scaled dataframes '''
    scaler = StandardScaler()
    scaled_split_dfs = copy.deepcopy(split_drug_dfs)
    for key in split_drug_dfs:
        scaled_x_train = scaler.fit_transform(scaled_split_dfs[key]['x_train'])
        scaled_split_dfs[key]['x_train'] = scaled_x_train
        scaled_x_test = scaler.transform(scaled_split_dfs[key]['x_test'])
        scaled_split_dfs[key]['x_test'] = scaled_x_test
    return scaled_split_dfs


def batch_pca(scaled_split_dfs, threshold):
    ''' Performs PCA resulting in a number of features with cumulative sum of
    explained covariance to the set threshold value. Threshold value should be
    between zero and one. Returns a dictionary with features and a dictionary
    with pca statistics'''
    pca = PCA(threshold)
    pca_dfs = copy.deepcopy(scaled_split_dfs)
    pca_ncomponent = {}
    pca_explained_variance = {}
    pca_explained_variance_ratios = {}
    pca_components = {}
    pca_stats = {}
    for key in pca_dfs:
        pca_x_train = pca.fit_transform(pca_dfs[key]['x_train'])
        pca_dfs[key]['x_train'] = pca_x_train
        pca_x_test = pca.transform(pca_dfs[key]['x_test'])
        pca_dfs[key]['x_test'] = pca_x_test
        pca_ncomponent[key] = pca.n_components_
        pca_components[key] = pca.components_
        pca_exp_var = pca.explained_variance_ratio_.cumsum().max().round(3)
        pca_explained_variance[key] = pca_exp_var
        pca_explained_variance_ratios[key] = pca.explained_variance_ratio_
    pca_stats['ncomponents'] = pca_ncomponent
    pca_stats['explained_var'] = pca_explained_variance
    pca_stats['components'] = pca_components
    pca_stats['explained_var_ratio'] = pca_explained_variance_ratios
    return pca_dfs, pca_stats


def batch_minibatch_sparse_pca(scaled_split_dfs, n_components, batch=50):
    ''' Performs minibatch sparse pca for each subset in dictionary of x and
    y train, and x and y test. Number of resulting components is set by
    n_components. For best results, n_compnents should be smaller than
    the number of samples. Batch determines how many features are analyzed at a
    time. Returns two dictionaries, one with the sparse pca
    features an done with information about the sparse pca done. '''
    sparse_pca_dfs = copy.deepcopy(scaled_split_dfs)
    sparse_mb_pca = MiniBatchSparsePCA(n_components=n_components,
                                       batch_size=batch, random_state=0)
    sparse_pca_ncomponents = {}
    sparse_pca_stats = {}
    for key in sparse_pca_dfs:
        sparse_mb_pca.fit(sparse_pca_dfs[key]['x_train'])
        sparse_pca_x_train = sparse_mb_pca.transform(sparse_pca_dfs[key]
                                                     ['x_train'])
        sparse_pca_dfs[key]['x_train'] = sparse_pca_x_train
        sparse_pca_x_test = sparse_mb_pca.transform(scaled_split_dfs[key]
                                                    ['x_test'])
        sparse_pca_dfs[key]['x_test'] = sparse_pca_x_test
        sparse_pca_ncomponents[key] = sparse_pca_x_train.shape[1]
    sparse_pca_stats['ncomponents'] = sparse_pca_ncomponents
    return sparse_pca_dfs, sparse_pca_stats


def pca_then_log_reg(scaled_split_dfs, threshold):
    ''' Using previously defined functions, performes PCA and
    then a logistic regression '''
    pca_dfs, pca_stats = batch_pca(scaled_split_dfs, threshold)
    y_preds, m_score, log_loss_metric, eval_dataframe = run_logreg(pca_dfs,
                                                                   0.01)
    return pca_dfs, eval_dataframe


def reduce_columns_by_stdev(split_dfs, remaining_percent_columns):
    ''' Limits the amount of features by retaining the columns with the highest
    standard deviation. How many columns is determined by remaining
    percent of columns. '''
    stdev_lim_dfs = copy.deepcopy(split_dfs)
    for key in split_dfs:
        genes_sorted_by_stdev = (split_dfs[key]
                                 ['x_test'].std(axis=0).
                                 sort_values(ascending=False).index)
        genes_keep = genes_sorted_by_stdev[0:(round
                                              (len(genes_sorted_by_stdev) *
                                               remaining_percent_columns))]
        stdev_lim_dfs[key]['x_test'] = split_dfs[key]['x_test'][genes_keep]
        stdev_lim_dfs[key]['x_train'] = split_dfs[key]['x_train'][genes_keep]
    return stdev_lim_dfs


def determine_correlation_coefficients(split_dfs):
    ''' Determines the correlation coefficient for each feature (gene) with
    the vital status outcome. Returns a dictionary with a correlation
    coefficient for each gene '''
    correlation_coefficients = {}
    for key in split_dfs:
        correlation_coefficients[key] = {}
        for column in split_dfs[key]['x_train'].columns:
            corr_coeff = (split_dfs[key]['x_train']
                          [column].corr(split_dfs[key]['y_train']))
            correlation_coefficients[key][column] = corr_coeff
    return correlation_coefficients


def reduce_columns_by_correlation_coefficients(split_dfs,
                                               corr_coeffs,
                                               remaining_percent_columns):
    ''' Limits the amount of features by retaining the columns with the highest
    correlation coefficient to the vital status. How many columns is determined
    by remaining percent of columns. '''
    corr_coeff_lim_dfs = copy.deepcopy(split_dfs)
    for key in corr_coeffs:
        sorted_correlation_coeff_per_drug = sorted(corr_coeffs[key].items(),
                                                   key=lambda x: abs(x[1]),
                                                   reverse=True)
        # drop all nan values
        genes_in_order = [gene[0] for gene in
                          sorted_correlation_coeff_per_drug if not
                          isnan(gene[1])]
        genes_keep = genes_in_order[0:(round
                                       (len(sorted_correlation_coeff_per_drug)
                                        * remaining_percent_columns))]
        corr_coeff_lim_dfs[key]['x_test'] = split_dfs[key][
                                                      'x_test'][genes_keep]
        corr_coeff_lim_dfs[key]['x_train'] = split_dfs[key][
                                                       'x_train'][genes_keep]
    return corr_coeff_lim_dfs


def reduction_metric_optimization(split_dfs, reduction_strategy='nan',
                                  percent_remaining=[1],
                                  correlation_coefficients='nan',
                                  scale='nan',
                                  pca='nan', threshold=[1],
                                  n_components=[5], c=0.001):
    ''' Using previously determined functions, allows the user to efficiently
     try multiple percent remaining conditions for their reduction of features
     using standard deviation or correlation coefficients. Scaling afterwards
     can be done as min max or standard scale, and can be followed by pca or
     sparse pca at various threshhold levels or number of components. '''
    evals_dataframes = {}
    pca_stats_total = {}
    for value in percent_remaining:
        if reduction_strategy == 'stdev':
            reduced_dataframe = reduce_columns_by_stdev(split_dfs, value)
        elif reduction_strategy == 'corr':
            reduced_dataframe = (reduce_columns_by_correlation_coefficients
                                 (split_dfs, correlation_coefficients, value))
        elif reduction_strategy == 'nan':
            reduced_dataframe = split_dfs
        if scale == 'min_max':
            scaled_dataframe = batch_min_max_scale(reduced_dataframe)
            final_dataframe = scaled_dataframe
        elif scale == 'standard':
            scaled_dataframe = batch_standard_scale(reduced_dataframe)
            final_dataframe = scaled_dataframe
        elif scale == 'nan':
            final_dataframe = reduced_dataframe
        if pca == 'nan':
            y_preds, m_score, log_loss, eval_dataframe = (run_logreg
                                                          (final_dataframe, c))
            evals_dataframes[f'{round(value*100)}_by_{reduction_strategy}_'
                             f'{scale}_scale'] = eval_dataframe
        elif pca == 'pca':
            for exp_variance in threshold:
                pca_stats, eval_dataframe = pca_then_log_reg(final_dataframe,
                                                             exp_variance)
                evals_dataframes[f'{round(value*100)}_by_{reduction_strategy}'
                                 f'_{scale}_'
                                 f'scale_pca_{exp_variance}'] = eval_dataframe
                pca_stats_total[f'{round(value*100)}_by_{reduction_strategy}_'
                                f'{scale}_scale_pca_'
                                f'{exp_variance}'] = pca_stats
        elif pca == 'sparse':
            for components in n_components:
                sparse_pca_dfs, sparse_pca_stats = (batch_minibatch_sparse_pca
                                                    (final_dataframe,
                                                     n_components=components,
                                                     batch=50))
                y_preds, m_score, log_loss, eval_dataframe = (run_logreg
                                                              (sparse_pca_dfs,
                                                               c))
                evals_dataframes[f'{round(value*100)}_by_{reduction_strategy}_'
                                 f'{scale}_scale_sparse_pca_'
                                 f'{components}'] = eval_dataframe
                pca_stats_total[f'{round(value*100)}_by_{reduction_strategy}_'
                                f'{scale}_scale_sparse_pca_'
                                f'{components}'] = sparse_pca_stats
    return evals_dataframes, pca_stats_total


def present_run_summary(all_results_dictionaries):
    ''' Takes a list of all evaluation dataframes and returns a dataframe
    listing the best method for each drug and evaluation method
    (ie accuracy, recall and etc.) '''
    list_of_results_dfs = []
    for eval_dict in all_results_dictionaries:
        list_of_results_dfs.append(pd.concat(eval_dict))
    summary_df = pd.concat(list_of_results_dfs)
    max_values = summary_df.unstack().max()
    best_methods = max_values.copy
    for combination in max_values.index:
        best_methods[combination] = summary_df.unstack()[summary_df.unstack()
                                                         [combination]
                                                         ==
                                                         summary_df.unstack()
                                                         [combination].max()]
    return best_methods

# limited_data_0624= pickle.load(open('limited_data_0624.pickle','rb'))
#
#
# # In[18]:
#
#
# df = one_hot_encode_vital_status(limited_data_0624)
#
#
# # In[19]:
#
#
# drugs_of_interest = df.standard_drugs.value_counts()[0:3].index.tolist()
#
#
# # In[20]:
#
#
# df_split = split_dataframe_by_drug(drugs_of_interest, df)
#
#
# # In[289]:
#
#
# df_split['gemcitabine']
#
#
# # In[22]:
#
#
# split_drug_dfs = batch_train_test_split(df_split)
#
#
# # In[23]:
#
#
# all_metrics_concat = pd.concat(allmetrics.values(), keys =allmetrics.keys())
#
#
# # In[24]:
#
#
# all_metrics_concat.unstack().sort_values(('accuracy', 'gemcitabine'), ascending=False).stack()
#
#
# # In[25]:
#
#
# evals_stdev_no_pca, no_pca = reduction_metric_optimization(split_drug_dfs,
#                                                       reduction_strategy='stdev',percent_remaining_list= [0.05, 0.1, 0.25, 0.5, 0.75])
#
#
# # In[26]:
#
#
# evals_stdev_no_pca
#
#
# # In[27]:
#
#
# pd.concat(evals_stdev_no_pca)
#
#
# # In[28]:
#
#
# correlation_coefficients = determine_correlation_coefficients(split_drug_dfs)
#
#
# # In[29]:
#
#
# evals_corr_no_pca, nan = reduction_metric_optimization(split_drug_dfs, reduction_strategy = 'corr', percent_remaining_list= [0.05, 0.1, 0.25, 0.5, 0.75], correlation_coefficients=correlation_coefficients)
#
#
# # In[30]:
#
#
# evals_corr_no_pca.keys()
#
#
# # In[31]:
#
#
# corr_coeff_followed_by_pca_evals, corr_coeff_pca_stats = reduction_metric_optimization(split_drug_dfs, reduction_strategy='corr', percent_remaining_list= [0.05, 0.1, 0.25, 0.5, 0.75], correlation_coefficients=correlation_coefficients, scale = 'standard', pca = 'pca', threshold = [0.01, 0.2, 0.5, 0.9] )
#
#
# # In[32]:
#
#
# corr_coeff_followed_by_pca_evals
#
#
# # In[33]:
#
#
# #just pd.DataFrame doesn't work
# corr_coeff_pca= pd.concat(corr_coeff_followed_by_pca_evals)
#
#
# # In[34]:
#
#
# corr_coeff_pca
#
#
# # In[35]:
#
#
# minibatch_bigfunc_evals, minibatch_bigfunc_pca_stats = reduction_metric_optimization(split_drug_dfs, reduction_strategy='nan',  scale = 'standard', pca = 'sparse', n_components=[15, 10, 5, 2])
#
#
# # In[36]:
#
#
# minibatch_bigfunc_evals
#
#
# # In[37]:
#
#
# all_metrics = {**evals_stdev_no_pca, **evals_corr_no_pca, **corr_coeff_followed_by_pca_evals, **minibatch_bigfunc_evals}
#
#
# # In[38]:
#
#
# all_metrics_df = pd.concat(all_metrics)
#
#
# # In[39]:
#
#
# all_metrics_df.unstack()
#
#
# # In[64]:
#
#
# evals_corr_no_pca
#
#
# # In[58]:
#
#
# test_dict_list = [evals_stdev_no_pca, evals_corr_no_pca, corr_coeff_followed_by_pca_evals, minibatch_bigfunc_evals]
#
#
# # In[59]:
#
#
# test_merge = (ChainMap(*test_dict_list))
#
#
# # In[74]:
#
#
# all_test_df = pd.concat(test_merge.maps[0])
#
#
# # In[78]:
#
#
# list_of_dfs=[]
# for item in test_dict_list:
#     list_of_dfs.append(pd.concat(item))
#
#
# # In[79]:
#
#
# list_of_dfs[0]
#
#
# # In[81]:
#
#
# test_summ = pd.concat(list_of_dfs)
#
#
# # In[82]:
#
#
# test_summ
#
#
# # In[88]:
#
#
# test_summ['accuracy'].groupby(level=1).max()['doxorubicin']
#
#
# # In[90]:
#
#
# test_summ['accuracy'][test_summ['accuracy']==test_summ['accuracy'].groupby(level=1).max()['doxorubicin']]
#
#
# # In[144]:
#
#
# max_values_test = test_summ.unstack().max()
#
#
# # In[145]:
#
#
# max_values_test
#
#
# # In[118]:
#
#
# best_methods = test_summ.unstack()[test_summ.unstack()[('score','doxorubicin')]==test_summ.unstack()[('score','doxorubicin')].max()].index
#
#
# # In[119]:
#
#
# best_methods
#
#
# # In[124]:
#
#
# max_values_test[('score','doxorubicin')]
#
#
# # In[138]:
#
#
# max_values_test[('score', 'doxorubicin')]=best_methods.to_list()
#
#
# # In[143]:
#
#
# max_values_test.unstack()
#
#
# # In[95]:
#
#
# test_summ.unstack()
#
#
# # In[266]:
#
#
# func_test = present_run_summary(test_dict_list)
#
#
# # In[271]:
#
#
# summary_df.unstack()[summary_df.unstack()[test_combo]==1]
#
#
# # In[226]:
#
#
# methods_dataframe = pd.DataFrame(func_test)
#
#
# # In[229]:
#
#
# methods_dataframe['values']= max_values
#
#
# # In[275]:
#
#
# methods_dataframe.loc[test_combo]['values']
#
#
# # In[284]:
#
#
# max_values = pd.DataFrame(summary_df.unstack().max(), columns=['values'])
#
#
# # In[287]:
#
#
# max_values.loc[test_combo,'values']
#
#
# # In[288]:
#
#
# def present_run_summary (all_results_dictionaries):
#     list_of_results_dfs=[]
#     for eval_dict in all_results_dictionaries:
#         list_of_results_dfs.append(pd.concat(eval_dict))
#     summary_df = pd.concat(list_of_results_dfs)
#     max_values = pd.DataFrame(summary_df.unstack().max(), columns = ['values'])
#     for combination in max_values.index:
#         best_method = summary_df.unstack()[summary_df.unstack()[combination]==max_values.loc[combination, 'values']].index.to_list()
#         max_values.loc[combination, 'method'] = best_method
#     return max_values
#
#
# # In[191]:
#
#
# list_of_results_dfs = []
# for eval_dict in test_dict_list:
#     list_of_results_dfs.append(pd.concat(eval_dict))
# summary_df = pd.concat(list_of_results_dfs)
#
#
# # In[192]:
#
#
# summary_df
#
#
# # In[193]:
#
#
# max_values = summary_df.unstack().max()
#
#
# # In[194]:
#
#
# max_values
#
#
# # In[195]:
#
#
# best_methods = max_values.copy()
#
#
# # In[218]:
#
#
# best_methods[test_combo]=summary_df.unstack()[summary_df.unstack()[test_combo]==max_values[test_combo]].index.to_list()
#
#
# # In[219]:
#
#
# best_methods
#
#
# # In[197]:
#
#
# test_combo= max_values.index[0]
#
#
# # In[202]:
#
#
# max_values[test_combo]
#
#
# # In[198]:
#
#
# test_combo
#
#
# # In[208]:
#
#
# summary_df.unstack()[summary_df.unstack()[test_combo]==max_values[test_combo]].index.to_list()
#
#
# # In[ ]:
