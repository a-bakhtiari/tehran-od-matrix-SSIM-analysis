# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, DBSCAN \
,MiniBatchKMeans, MeanShift, SpectralClustering
from sklearn.metrics import silhouette_score
import pandas as pd
from google.colab import files
dataframe = pd.read_excel('SE1393.xlsx')#read the zone file
dataframe.dropna(inplace=True)
dataframe.rename(columns={'commercial_unit': 'commercial_landuse'}, inplace=True)
dataframe

# needed these variables
variable_lists =['pop',	'emp_pop', 'veh_own',
        'karmnd_dr_mhl_shghl',	'office_land_use',
        	'commercial_landuse']
          
dataframe[variable_lists] = dataframe[variable_lists]

# normalizing data

dataframe[variable_lists] = dataframe[variable_lists].apply(lambda x: (x - x.min()) / (x.max()-x.min())) #
X = dataframe[variable_lists].values

import itertools

list_combinations = list()
for L in range(0, len(variable_lists)+1):
    for subset in itertools.combinations(variable_lists, L):
      if len(subset)>= 2:
        list_combinations.append(list(subset))

dataframe

def make_cluster(list_combin):
  #features
  X = dataframe[list_combin].values
  for i in range(2,6):
    Birch_model = Birch(threshold=0.01, n_clusters=i)
    SpectralClustering_model = SpectralClustering(n_clusters=i)
    kmeans = KMeans(n_clusters=i, n_init=20, n_jobs=4)
    mkmeans = MiniBatchKMeans(n_clusters=i)
    agg_model = AgglomerativeClustering(n_clusters=i)

    y_pred_Birch = Birch_model.fit_predict(X)
    y_pred_SpectralClustering = SpectralClustering_model.fit_predict(X)
    y_pred_kmeans = kmeans.fit_predict(X)
    y_pred_mkmeans = mkmeans.fit_predict(X)
    y_pred_agg = agg_model.fit_predict(X)

    col_name_b = "Birch_"+ str(i)
    col_name_s = "SpectralClustering_"+ str(i)
    col_name_k = "KMeans_"+ str(i)
    col_name_mk = "mkmeans_"+ str(i)
    col_name_agg = "agg_model_"+ str(i)

    dataframe[col_name_s] = y_pred_SpectralClustering
    dataframe[col_name_b] = y_pred_Birch
    dataframe[col_name_k] = y_pred_kmeans
    dataframe[col_name_mk] = y_pred_mkmeans
    dataframe[col_name_agg] = y_pred_agg

  list_b = []
  list_s = []
  list_k = []
  list_mk = []
  list_agg = []


  for i in range(2,6):
    col_name_b = "Birch_"+ str(i)
    col_name_s = "SpectralClustering_"+ str(i)
    col_name_k = "KMeans_"+ str(i)
    col_name_mk = "mkmeans_"+ str(i)
    col_name_agg = "agg_model_"+ str(i)

    value_b = silhouette_score(X, dataframe[col_name_b])
    value_s = silhouette_score(X, dataframe[col_name_s])
    value_k =silhouette_score(X, dataframe[col_name_k])
    value_mk =silhouette_score(X, dataframe[col_name_mk])
    value_agg =silhouette_score(X, dataframe[col_name_agg])

    list_b.append(value_b)
    list_s.append(value_s)
    list_k.append(value_k)
    list_mk.append(value_mk)
    list_agg.append(value_agg)


  my_python_list = [list_b, list_s ,list_k ,list_mk ,list_agg]
  new_df = pd.DataFrame(columns=['n=2', 'n=3','n=4','n=5'], data=my_python_list)

  Data = {'birch': list_b,
        'spectral': list_s,
        'kmeans': list_k,
        'mkmeans': list_mk,
        'agg':list_agg,
        'number':['2','3','4','5']
       }
  
  df = pd.DataFrame(Data,columns=['birch','spectral','kmeans','mkmeans','agg','number'])
  import matplotlib.pyplot as plt
  from matplotlib.font_manager import FontProperties
  
  fontP = FontProperties()
  fontP.set_size('large')
 
  plt.plot(df['number'], df['birch'],color='black', marker="P", label='Birch')
  plt.plot(df['number'], df['spectral'], color='black', marker="^",label='Spectral',dashes=[3, 3])
  plt.plot(df['number'], df['kmeans'], color='black', marker="v",label='K-means',dashes=[5, 5])
  plt.plot(df['number'], df['mkmeans'], color='black', marker="x",label='Mini batch K-means',dashes=[10, 10])
  plt.plot(df['number'], df['agg'], color='black', marker='o',label='Agglomerative', dashes=[1, 1])
  plt.xlabel('Number of clusters', fontsize=14)
  plt.ylabel('Silhouette Coef???cient', fontsize=14)
  plt.legend(prop={'size': 11})
  plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
  plt.grid(True)

  name = str(list_combin)

  plt.savefig(name  + '.png' ,bbox_inches='tight')
  plt.close()

  df_value_counts = pd.DataFrame()
  for i in range(2,6):
    col_name_b = "Birch_"+ str(i)
    col_name_s = "SpectralClustering_"+ str(i)
    col_name_k = "KMeans_"+ str(i)
    col_name_mk = "mkmeans_"+ str(i)
    col_name_agg = "agg_model_"+ str(i)

    valuecount_b = dataframe[col_name_b].value_counts().sort_index().rename("Birch")
    valuecount_s = dataframe[col_name_s].value_counts().sort_index().rename("SpectralClustering")
    valuecount_k =dataframe[col_name_k].value_counts().sort_index().rename("KMeans")
    valuecount_mk =dataframe[col_name_mk].value_counts().sort_index().rename("mkmeans")
    valuecount_agg =dataframe[col_name_agg].value_counts().sort_index().rename("agg_model")

    d = {'Birch': '-', 'SpectralClustering': '-', 'KMeans': '-','mkmeans': '-','agg_model': '-',}

    df7 = pd.concat([valuecount_b,valuecount_s,valuecount_k,valuecount_mk,valuecount_agg], axis=1)
    dfx = pd.DataFrame(data=d,index=['x'])
    df_value_counts = pd.concat([df_value_counts,  df7, dfx], axis=0)
  df_value_counts.to_csv(name + '_value_count '+'.csv')
  dataframe.to_csv(name+'.csv')

for combination in list_combinations:
  make_cluster(combination)
