import pandas as pd
import numpy as np

import numbers

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score

from sklearn.feature_selection import VarianceThreshold

seed = 2022


def combine_low_freq_categories(feature, percentage_thres = 5):

   # docstring

   feature = pd.Series(feature)
   series = feature.value_counts()
   mask = (series/series.sum() * 100).lt(percentage_thres)

   return np.where(feature.isin(series[mask].index),'Other',feature)

def make_buckets(feature, no_buckets = 6, percentage_thres = 5, as_array=False):

   """
   Transform continuous numeric feature in bucketed feature.
   Args:
       feature (Pandas Series, numpy Array or list): 
           Continuous numeric feature that needs to be bucketed
       no_buckets (int, optional): 
           Number of buckets into which feature will be divided.
           If the feature has any missing value, they will be
           assigned to additional bucket.
           Defaults to 6.
       percentage_thres (number, optional): 
           Percentage threshold for categorical features.
           All categories below this threshold will be
           combined into one category - 'Other'
           Defaults to 5.    
       as_array (bool, optional): 
           Should the output be array? If set to False, 
           the function returns Pandas Series. 
           Defaults to False.
   Returns:
       Pandas Series (or Array).
   """

   # Todo
   # co jesli wartosci jest mniej niz no_buckets?

   # Checks
   if type(feature) is not np.ndarray and type(feature) is not pd.Series and type(feature) is not list:
       raise TypeError('`feature` must be either list, numpy array or pandas Series.')

   if type(no_buckets) is not int:
       raise TypeError('`no_buckets` must be a positive integer.')

   if no_buckets <= 0:
       raise ValueError('`no_buckets` must be a positive integer.')

   # Divide into buckets
   # if categorical feature all categories below 5% are merged into 'Other' category
   if pd.Series(feature).dtype == object:
       bucket_vec = combine_low_freq_categories(feature, percentage_thres = percentage_thres)
   else:
       bucket_vec = pd.qcut(feature, no_buckets, labels=False, duplicates = 'drop')
       bucket_vec = pd.Series(bucket_vec).fillna(-1)

   # Return numpy array?
   if as_array:
       bucket_vec = np.array(bucket_vec)

   return bucket_vec

def count_woe(feature, label, correction = 0.5, start_with_buckets = False, *args, **kwargs):

   """
   Count Weight of Evidence (WoE) for a given feature.
   Args:
       feature (Pandas Series, numpy Array or list): 
           Feature for which WoE will be computed.
       label (Pandas Series, numpy Array or list):
           Target variable that will be used to compute WoE.
       correction (numeric, optional): 
           Number that ensures that WoE is computed even if
           there are no cases of one class from target variable
           available in a bucket.
           Defaults to 0.5.
       start_with_buckets (bool, optional): 
           Is feature already bucketed? If no, then first it
           will be bucketed using make_buckets function.
           Defaults to False.
       args: Passed to make_buckets.
       kwargs: Passed to make_buckets.
   Returns:
       Returns a Pandas DataFrame with information of
       WoE in each bucket, as well as number of observations
       and share of each target type in all buckets.
   """

   # Checks
   if type(feature) is not np.ndarray and type(feature) is not pd.Series and type(feature) is not list:
       raise TypeError('`feature` must be either list, numpy array or pandas Series.')

   if type(label) is not np.ndarray and type(label) is not pd.Series and type(label) is not list:
       raise TypeError('`feature` must be either list, numpy array or pandas Series.')

   if len(feature) != len(label):
       raise ValueError('`feature` and `label` must have the same lengths.')

   # if type(correction) is not int and type(correction) is not float:
   if not isinstance(correction, numbers.Number):    
       raise TypeError('`correction` must be a positive numeric.')

   if correction <= 0:
       raise ValueError('`correction` must be a positive numeric.')

#     if type(no_buckets) is not int:
#         raise TypeError('`no_buckets` must be a positive integer.')

#     if no_buckets <= 0:
#         raise ValueError('`no_buckets` must be a positive integer.')

   if not start_with_buckets:
       feature = make_buckets(feature = feature, *args, **kwargs)

   woe_df = pd.DataFrame({'bucket': feature, 'status': label})
   all_no = feature.size
   all_bad = np.sum(label)
   all_good = feature.size - all_bad


   result_df = woe_df.groupby('bucket', as_index=False) \
       .agg(
           no_all = ('status', 'count'), 
           no_bad = ('status', 'sum')
       )

   result_df['no_good'] = result_df['no_all'] - result_df['no_bad']
   result_df['share_good'] = (result_df['no_good'] + correction)/ (all_good + correction)
   result_df['share_bad'] = (result_df['no_bad'] + correction)/ (all_bad + correction)
   result_df['woe'] = np.log(result_df['share_good']/result_df['share_bad'])

   conditions = [(result_df['no_good'] > 0) & (result_df['no_bad'] > 0), (result_df['no_good'] == 0) | (result_df['no_bad'] == 0)]
   values = [np.sqrt((1/result_df['no_good']) + (1/result_df['no_bad']) - (1/all_good) - (1/all_bad)), 0]

   result_df['std_woe'] = np.select(conditions, values) * 2

   # result_df['std_woe'] = np.sqrt((1/result_df['no_good']) + (1/result_df['no_bad']) - (1/all_good) - (1/all_bad))

   return result_df

def plot_woe(df, group_col_name = 'bucket', woe_col_name='woe'):

   # przedzialy jako nazwy bucketow
   # docstring

   sns.set_style('white')
   custom_palette = {}
   for q in set(df[group_col_name]):
       woe = df.loc[df[group_col_name] == q, woe_col_name]
       if (woe < 0).bool():
           custom_palette[q] = '#D82C20'
       else:
           custom_palette[q] = '#99E472' 

   yerr = [df['std_woe'], df['std_woe']]

   g = sns.catplot(x=group_col_name, y=woe_col_name, kind='bar',data=df, legend=None, palette=custom_palette)
   plt.errorbar(x=df.index, y=df['woe'], yerr=yerr, fmt='none', c='r')

   g.fig.set_size_inches(15, 8)        


def count_iv(feature, label, correction = 0.5, start_with_buckets = False, *args, **kwargs):

   """
   Count Information Value (IV) for a given feature.
   Args:
       feature (Pandas Series, numpy Array or list): 
           Feature for which WoE will be computed.
       label (Pandas Series, numpy Array or list):
           Target variable that will be used to compute IV.
       correction (numeric, optional): 
           Number that ensures that IV is computed even if
           there are no cases of one class from target variable
           available in a bucket.
           Defaults to 0.5.
       start_with_buckets (bool, optional): 
           Is feature already bucketed? If no, then first it
           will be bucketed using make_buckets function.
           Defaults to False.
       args: Passed to make_buckets.
       kwargs: Passed to make_buckets.
   Returns:
       Returns single numeric value of IV for a given feature.
   """

   # Checks
   if type(feature) is not np.ndarray and type(feature) is not pd.Series and type(feature) is not list:
       raise TypeError('`feature` must be either list, numpy array or pandas Series.')

   if type(label) is not np.ndarray and type(label) is not pd.Series and type(label) is not list:
       raise TypeError('`feature` must be either list, numpy array or pandas Series.')

   if type(correction) is not int and type(correction) is not float:
       raise TypeError('`no_buckets` must be a positive numeric.')

   if len(feature) != len(label):
       raise ValueError('`feature` and `label` must have the same lengths.')

   if correction <= 0:
       raise ValueError('`correction` must be a positive numeric.')

#     if type(no_buckets) is not int:
#         raise TypeError('`no_buckets` must be a positive integer.')

#     if no_buckets <= 0:
#         raise ValueError('`no_buckets` must be a positive integer.')

   if not start_with_buckets:
     feature = make_buckets(feature = feature,  *args, **kwargs)

   woe_df = count_woe(feature, label, correction=correction, start_with_buckets=True)
   iv = np.sum((woe_df['share_good'] - woe_df['share_bad']) * woe_df['woe']) 

   return iv

def count_iv_for_df(df, label_name, correction=0.5, start_with_buckets=False, include_only = None, exclude = None, *args, **kwargs):

   """
   Generates Information Values (IV) for all features in data frame
   given specified target variable.
   Args:
       df (Pandas Data Frame): 
           Table which contains features and target variable
       label_name (string): 
           Name of targer variable.
       correction (numeric, optional): 
           Number that ensures that IV is computed even if
           there are no cases of one class from target variable
           available in a bucket.
           Defaults to 0.5.
       start_with_buckets (bool, optional): 
           Is feature already bucketed? If no, then first it
           will be bucketed using make_buckets function.
           This arguments only applies to numeric features.
           For categorical features each category is treated
           as separate bucket. 
           Defaults to False.
       include_only (list, optional):
           List of features for which IV should be computed.
           Defaults to None.
       exclude (list, optional):
           List of features for which IV should not be computed.
           Defaults to None.
       args: Passed to count_iv.
       kwargs: Passed to count_iv.
   Returns:
       Pandas Data Frame with 2 columns (arranged by IV):
           - feature: name of the feature,
           - iv_value: Information Value of a feature.
   """

   # dodac wybor zmiennych ktore chcemy policzyc DONE
   # albo tych ktorych nie chcemy DONE
   # ograniczyc liczbe bucketow dla zmiennych kategorycznych DONE
   # co jesli jest mniej wartosci niz bucketow?

   # Checks
   if type(df) is not pd.DataFrame:
       raise TypeError('`data` must be Pandas Data Frame.')

   if type(label_name) is not str:
       raise TypeError('`label_name` must be a string.')

   if not isinstance(correction, numbers.Number):    
       raise TypeError('`correction` must be a positive numeric.')

   if correction <= 0:
       raise ValueError('`correction` must be a positive numeric.')

   data = df

   if include_only is not None:
       if type(include_only) is not list:
           raise TypeError('`include_only` must be a list.')

       include_only.append(label_name)
       data=df[include_only] 

   if exclude is not None:
       if type(exclude) is not list:
           raise TypeError('`exclude` must be a list.')

       data = df.drop(exclude, axis=1) 

   df_feat_num = data.drop(label_name, axis=1).select_dtypes(np.number)
   df_feat_char = data.drop(label_name, axis=1).select_dtypes(object)
   label = data[label_name]

   if start_with_buckets:
       df_res_num = df_feat_num.apply(lambda x: count_iv(x, label, correction=correction, start_with_buckets=True))
       df_res_char = df_feat_char.fillna('missing_value').apply(lambda x: count_iv(x, label, correction=correction, start_with_buckets=True))
   else:  
       df_res_num = df_feat_num.apply(lambda x: count_iv(x, label, correction=correction, *args, **kwargs))
       df_res_char = df_feat_char.apply(lambda x: make_buckets(x, *args, **kwargs)) \
           .fillna('missing_value') \
           .apply(lambda x: count_iv(x, label, correction=correction, start_with_buckets=True))

   # df_res_char = df_feat_char.fillna('missing_value').apply(lambda x: count_iv(x, label, correction=correction, start_with_buckets=True))

   if not isinstance(df_res_num, pd.Series):
       df_res_num = pd.DataFrame()

   if not isinstance(df_res_char, pd.Series):
       df_res_char = pd.Series(dtype='float')

   results_df = pd.concat([df_res_num, df_res_char]).reset_index()
   results_df.columns =['feature', 'iv_value']
   results_df = results_df.sort_values('iv_value', ascending=False)

   return results_df


def get_feats(df, feats=[''], exclude_feats=True): 
   """
   Get the list of features.
   Parameters
   ----------
   df : DataFrame
       Dataframe to prepare features
   feats : list
       List of features to exclude or include from dataframe
   exclude_feats : bool
       if true then feats list is excluding columns, if True is including only columns from feat list
   Returns
   -------
   feats : list
       Final List of features
   """
   if exclude_feats:
     final_feats = [f for f in df.columns if f not in feats]
   else:
     final_feats = [f for f in df.columns if f in feats]
   return final_feats

def get_X(df, feats=[''], exclude_feats=True, output_df=True): 
   """
   Prepare dataframe base od features list.
   Parameters
   ----------
   df : DataFrame
       Dataframe with all features and columns
   feats : list
       List of features to exclude or include from dataframe
   exclude_feats : bool
       if true then feats list is excluding columns, if True is including only columns from feat list
   output_df : bool
       if true return dataframe else array
   Returns
   -------
   feats : DataFrame
       Final DataFrame without features to exclude
   """    

   if output_df:
     return df[ get_feats(df, feats, exclude_feats=exclude_feats) ]
   else:
     return df[ get_feats(df, feats, exclude_feats=exclude_feats) ].values


from sklearn.feature_selection import VarianceThreshold

def selectionVarianceThreshold(df, variance_threshold=0.95):
   sel = VarianceThreshold(threshold=(1 - variance_threshold))
   sel_var=sel.fit_transform(df)
   sel_loc_index = sel.get_support(indices=True)

   if len(sel_loc_index)==0:
     return None
   else: 
     return list(df.columns[sel_loc_index])

from sklearn.ensemble import RandomForestClassifier

def add_random_variables(dataset, num_rand_vars=10):
   """Function adds random variables to dataset.

   Args:
       dataset, pd.DataFrame: DataFrame, where columns are features.
       num_rand_vars,int: Numbers of random variables to add.

   Returns:
       pd.DataFrame: Dataset with random variables added.
   """
   rand_vars_names = ['random_' + str(i) for i in range(num_rand_vars)]
   random_vars = pd.DataFrame(np.random.rand(len(dataset), num_rand_vars),
                              columns=rand_vars_names)
   random_cols = random_vars.columns

   return pd.concat([dataset, random_vars.set_index(dataset.index)], axis=1), list(random_cols)

def selectionRandomForestThreshold(X_train, y_train, X_val, y_val, num_rand_vars=20, user_rf_params = {'n_estimators': 500, 'max_depth': 6}):
 X_train_tmp, random_cols = add_random_variables(X_train, num_rand_vars=num_rand_vars)

 model_rf = RandomForestClassifier(**user_rf_params)
 model_rf.fit(X_train_tmp, y_train)

 features = pd.concat([pd.DataFrame(model_rf.feature_importances_), pd.DataFrame(X_train_tmp.columns)], axis = 1)
 features.columns = ['importance', 'char_name']
 features = features.sort_values(by='importance', ascending = False)

 importance_cut_off = features[features['char_name'].isin(random_cols)]['importance'].mean()

 return list(features[(features['importance'] >= importance_cut_off) & (~features['char_name'].isin(random_cols))]['char_name'])


def select_top_features(model, X, n_chars=1):
 try:
   features = pd.concat([pd.DataFrame(model.feature_importances_), pd.DataFrame(X.columns)], axis = 1)
   features.columns = ['importance', 'char_name']
   features = features.sort_values(by='importance', ascending = False)
   features_short = features['char_name'][0:n_chars].to_list()
 except:
   features_short=None

 return features_short

def selectionXGBoostTopX(X_train, y_test, X_val, y_val, n_step=5, 
                        user_xgb_params = {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 250, 'seed': seed, 'subsample': 0.75, 'colsample_bytree': 0.75, 'colsample_bylevel': 0.75,}):
 model_all = xgb.XGBClassifier(**user_xgb_params).fit(X_train, y_train)

 if X_train.shape[0] >= n_step:
   i = 1
   metric_champion = 0
   feats_champion = ['']

   while X_train.shape[0] >= n_step*i:
     feats = select_top_features(model_all, X_train, n_chars=n_step*i)
     summary_challenger, model_challenger = model_new_feats_calculation(X_train, y_train, X_val, y_val, feats_to_include=feats)
     metric_challenger = summary_challenger.loc['VAL']['f1']

     if metric_challenger > metric_champion:
       metric_champion = metric_challenger
       feats_champion = feats
     else:
       break

     i = i+1
 else:
   print('brak tylu zmiennych')  

 return feats_champion

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# wybor top z najwiekszym prawdopodobienstwem
def metrics_percent_population_badRate(y, y_pred, q, top=True):
   tmp = pd.concat([pd.DataFrame(y), pd.DataFrame(y_pred)], axis = 1)
   tmp.columns = ['y','y_pred']

   if top:
     tmp_q = tmp[tmp['y_pred'] > tmp['y_pred'].quantile(1-q)]
   else :
     tmp_q = tmp[tmp['y_pred'] <= tmp['y_pred'].quantile(1-q)]

   return tmp_q['y'].sum() / tmp_q['y'].count()

#Funkcja zwracaąca wartości zdefiniowanych metrych dla przygotowanej próbki
def create_measures(y,y_pred,y_pred_bin): 
   '''Function calulate metrics based on target and predicted values.

   Args:
       y, array: target.
       y_pred, array: predicted target.
       y_pred_bin, array: predicted target as bin

   Returns:
       df_metrics, DataFrame: DataFrame with calculated metrics.
   '''
   score_test = roc_auc_score(y, y_pred)
   Gini_index = 2*score_test - 1

   confusion_matrix_main = confusion_matrix(y, y_pred_bin)
   TP = confusion_matrix_main[0][0]
   FP = confusion_matrix_main[0][1]
   FN = confusion_matrix_main[1][0]
   TN = confusion_matrix_main[1][1]

   accuracy = accuracy_score(y, y_pred_bin)
   precission = precision_score(y, y_pred_bin)
   recall = recall_score(y, y_pred_bin)
   f1 = f1_score(y, y_pred_bin)

   # tworzenie metryk jaki jest % targetów 1 w top x% populacji
   top_05 = metrics_percent_population_badRate(y, y_pred, 0.05, top=True)
   top_25 = metrics_percent_population_badRate(y, y_pred, 0.25, top=True)
   top_50 = metrics_percent_population_badRate(y, y_pred, 0.50, top=True)

   df_metrics = { 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
               'AUC': [round(score_test,4)], 'GINI': [round(Gini_index,4)],
               'accuracy': [round(accuracy,4)], 'precission': [round(precission,4)],
               'recall': [round(recall,4)],'f1': [round(f1,4)],
               #'top05': [round(top_05,4)],'top25': [round(top_25,4)],'top50': [round(top_50,4)],
                }

   df_metrics = pd.DataFrame.from_dict(df_metrics)

   return df_metrics

def calculating_metrics(model, X_train, y_train
                       , X_val=np.array([[0],[0]]), y_val=np.array([[0],[0]])):
   '''Function calulate probability and concate metrics to one DataFrame.

   Args:
       model, model: calculated clf model
       X_train, array: data on which model was calculated
       y_train, array: target
       X_val, array: data validation sample
       y_val, array: target on validation sample

   Returns:
       df_metrics, DataFrame: DataFrame with calculated metrics.
   '''    
   y_pred_train = model.predict_proba(X_train)[:, 1] #prawdopodobieństwa
   y_pred_train_bin = model.predict(X_train) #wartości 0/1
   summary_train = create_measures(y_train,y_pred_train,y_pred_train_bin)

   if X_train.shape[1] == X_val.shape[1]: #jeśli jest ta sama liczba kolumn
       y_pred_val = model.predict_proba(X_val)[:, 1]   
       y_pred_val_bin = model.predict(X_val)
       summary_val = create_measures(y_val,y_pred_val,y_pred_val_bin)

   if X_train.shape[1] == X_val.shape[1]:
       measures =  pd.concat([summary_train,summary_val]).set_index([pd.Index(['TRAIN', 'VAL'])])     
   else:
       measures = summary_train.set_index([pd.Index(['TRAIN'])])     

   return measures
