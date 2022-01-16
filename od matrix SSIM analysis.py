# -*- coding: utf-8 -*-
"""
#OD matrix
"""

#download OD_matrix data

import os
import zipfile

with zipfile.ZipFile("/content/OD_matrix.zip","r") as zip_ref:
    zip_ref.extractall("OD_matrix")
    
Dir = os.path.join('OD_matrix')
showlist = os.listdir(Dir)

# Observe the folder's content
print(showlist)

from datetime import datetime
import json
#calculate day of the week
def cal_day(showlist=showlist):
  n = 0
  week_dict= dict()
  for i in showlist:
    n += 1
    year = int(i[0:4])
    month = int(i[5:7])
    day = int(i[8:10])
    wk_day = datetime(year, month, day).strftime('%A')
    if wk_day not in week_dict.keys():
      week_dict[(wk_day)] = [i]
    else:
      week_dict[(wk_day)].append(i)
    
  with open('wk_day_dict.txt', 'w') as file:
     file.write(json.dumps(week_dict))
  return week_dict

#run the above function
week_day_dict = cal_day()

import pandas as pd
import numpy as np

#make a list of TAZ's
list_navahi = list()
for i in range(1,701):
  list_navahi.append(i)

def make_absent_taz(df, list_navahi, Origin='Origin', Destination='Destination'):
  main_list_o = np.setdiff1d(list_navahi, df[Origin].unique())
  main_list_d = np.setdiff1d(list_navahi, df[Destination].unique())

  #areas that are not present in the calculated day
  absent_areas = list(main_list_o) + list(main_list_d)

  # make a dataframe of absent areas and add it to the main dataframe
  df2 = pd.DataFrame(list(zip(absent_areas, absent_areas)), 
               columns =[Origin, Destination]) 
  df2['Count'] = 0

  df = pd.concat([df, df2], axis=0)
  df = df[df['Origin'] <= 700]
  df = df[df['Destination'] <= 700]  
  return df


def to_matrix(df, Origin_zone, Destination_zone, Count):
  # The key operation
  matrix = (
    df.pivot_table(index=Origin_zone, columns=Destination_zone, values=Count)
    .fillna(0)
    )
  return matrix

cols = ['Origin', 'Destination', 'Count']

def cal_sum_average(list_names, wk_day, type_cal):
  '''gets a list of names of the files in each day and the week day
  and return the average od matrix for each day as a dictionary'''
  for x in list_names:
    #assign the path for each file
    name = 'OD_matrix/' + x
    # make a dataframe
    df = pd.read_csv(name ,usecols=cols)

    if type_cal!='taz':
      #df['sh_o_agg'] = df['Origin'].copy()
      #df['sh_d_agg'] = df['Destination'].copy()
      df.dropna( inplace=True)

    #if type_cal=='taz': #if we are calculating based on taz create absent taz's
    df = make_absent_taz(df, list_navahi)
    #OD_matrix = to_matrix(df, 'Origin', 'Destination', 'Count')

    if type_cal=='sh':
      # needed to aggregate over simillar area code combinations to prevent huge loss of trips
      #df.reset_index(drop=True, inplace=True)
      #df = df.groupby(['sh_o_agg','sh_d_agg'])['Count'].agg(['sum'])
      #df = df.rename(columns={'sum':'Count'}).reset_index()
    
      #df = df.astype('int32')
      OD_matrix = to_matrix(df, 'Origin', 'Destination', 'Count')

    elif type_cal=='pah_slp':
      # shahrdari area to pahneh
      df['pah_o_agg'] = df['Origin'].map(pahneh_dict)
      df['pah_d_agg'] = df['Destination'].map(pahneh_dict)

      # slp classification
      df['slp_o_agg'] = df['Origin'].map(slp_dict)
      df['slp_d_agg'] = df['Destination'].map(slp_dict)
      df.dropna( inplace=True)

      df['pah_sh_slp_o_agg'] = df['pah_o_agg'].astype('str') +'_'+ (df['slp_o_agg'].astype('int32')).astype('str') \
                               +'_'+ (df['Origin'].astype('int32')).astype('str')

      df['pah_sh_slp_d_agg'] = df['pah_d_agg'].astype('str') +'_'+ (df['slp_d_agg'].astype('int32')).astype('str') \
                               +'_'+ (df['Destination'].astype('int32')).astype('str')

      # needed to aggregate over simillar areacode combinations to prevent huge loss of trips
      df.reset_index(drop=True, inplace=True)
      #df = df.groupby(['pah_sh_slp_o_agg','pah_sh_slp_d_agg'])['Count'].agg(['sum'])
      #df = df.rename(columns={'sum':'Count'}).reset_index()
      OD_matrix = to_matrix(df, 'pah_sh_slp_o_agg', 'pah_sh_slp_d_agg', 'Count')
    elif type_cal=='tab':
      # shahrdari area to datebandi
      df['tab_o_agg'] = df['Origin'].map(tabaghe_bandi_dict)
      df['tab_d_agg'] = df['Destination'].map(tabaghe_bandi_dict)
      df.dropna( inplace=True)
    
      df['tab_sh_o_agg'] = df['tab_o_agg'].astype('str') +'_'+ (df['Origin'].astype('int32')).astype('str') 

      df['tab_sh_d_agg'] = df['tab_d_agg'].astype('str') +'_'+ (df['Destination'].astype('int32')).astype('str')

      # needed to aggregate over simillar areacode combinations to prevent huge loss of trips
      df.reset_index(drop=True, inplace=True)
      #df = df.groupby(['tab_sh_o_agg','tab_sh_d_agg'])['Count'].agg(['sum'])
      #df = df.rename(columns={'sum':'Count'}).reset_index()
      OD_matrix = to_matrix(df, 'tab_sh_o_agg', 'tab_sh_d_agg', 'Count')

    if 'OD_matrix_sum' not in locals():
      OD_matrix_sum = OD_matrix.copy()
    else:
      OD_matrix_sum += OD_matrix
  sum = OD_matrix_sum
  average = sum/len(list_names)
  #save the output
  #average.to_csv(wk_day+'_avrage.csv')
  return average

taz_matrixes_dict_avg = dict()# a dictionary with 7 keys(day of the week).matrixes are averaged for each day(TAZ) 
for x in week_day_dict.keys():
  taz_matrixes_dict_avg[x] = cal_sum_average(week_day_dict[x], x, type_cal='taz')

"""# Aggregate TAZ's"""

# read key information that are going to replace taz
shardari_key = pd.read_excel('https://www.dropbox.com/scl/fi/jkcn7cv8krp257bqsawob/SE1393-TAZ-Navahi-Mantaghe.xlsx?dl=1&rlkey=pm973vqadncdguoov3lwgrt3v')
pahneh_key = pd.read_excel('https://www.dropbox.com/scl/fi/jes5z6t6kqxkf3elr2fs4/Subregion-Pahneh.xlsx?dl=1&rlkey=aziamyo2ierw6zm23t1g0prw9')

#there are two sheets so we have to seperate them before making the dataframe
xls = pd.ExcelFile('https://www.dropbox.com/scl/fi/dnz6opfauv56bh5mzehbk/tabagheh-bandi-navahi-finall.xlsx?dl=1&rlkey=ojm36q2zbwtx31sz2zbhunuzf')
tabaghe_bandi_key = pd.read_excel(xls, 'Sheet1')

#rename persian columns
shardari_key.rename(columns={'ناحیه شهرداری': 'nahie_shahrdari', 'ناحیه ترافیکی':'TAZ'}, inplace=True)
pahneh_key.rename(columns={'ناحیه شهرداری': 'nahie_shahrdari', 'پهنه بندی':'pahneh'}, inplace=True)
tabaghe_bandi_key.rename(columns={'ناحیه شهرداری': 'nahie_shahrdari', 'دسته بندی':'dastebandi'}, inplace=True)

#renaming clusters to be more recoganizable
tabaghe_bandi_key['dastebandi'].replace(1, 'cluster_1', inplace=True)
tabaghe_bandi_key['dastebandi'].replace(2, 'cluster_2', inplace=True)
tabaghe_bandi_key['dastebandi'].replace(3, 'cluster_3', inplace=True)
tabaghe_bandi_key['dastebandi'].replace(4, 'cluster_4', inplace=True)
tabaghe_bandi_key['dastebandi'].replace(5, 'cluster_5', inplace=True)

nahie_shahrdari_dict = dict(shardari_key[['TAZ', 'nahie_shahrdari']].values)

pahneh_dict = dict(pahneh_key[['nahie_shahrdari', 'pahneh']].values)
tabaghe_bandi_dict = dict(tabaghe_bandi_key[['nahie_shahrdari', 'dastebandi']].values)

#baraye pahneh bayad yek zir majmooeye digar ezafe konam
slp_dict = dict(pahneh_key[['nahie_shahrdari', 'Hight-Low SLP']].values)

shardari_key['pahneh'] = shardari_key['nahie_shahrdari'].map(pahneh_dict)
shardari_key['tabagheh'] = shardari_key['nahie_shahrdari'].map(tabaghe_bandi_dict)
shardari_key['slp'] = shardari_key['nahie_shahrdari'].map(slp_dict)

list_taz = list(shardari_key['TAZ'])

shardari_key.dropna( inplace=True)

pahneh_dict = dict(shardari_key[['TAZ', 'pahneh']].values)
tabaghe_bandi_dict = dict(shardari_key[['TAZ', 'tabagheh']].values)

#baraye pahneh bayad yek zir majmooeye digar ezafe konam
slp_dict = dict(shardari_key[['TAZ', 'slp']].values)

"""##Shardari"""

shahrdari_matrixes_dict_avg = dict()# a dictionary with 7 keys(day of the week).matrixes are averaged for each day(shahrdari) 
for x in week_day_dict.keys():
  shahrdari_matrixes_dict_avg[x] = cal_sum_average(week_day_dict[x], x, type_cal='sh')

"""##Pahneh + slp"""

pahneh_slp_matrixes_dict_avg = dict()# in columns: the first number from right is the sh area code and the next number is slp class 
for x in week_day_dict.keys():
  pahneh_slp_matrixes_dict_avg[x] = cal_sum_average(week_day_dict[x], x, type_cal='pah_slp')

"""##tabaghe bandi(daste bandi)"""

tabaghe_bandi_matrixes_dict_avg = dict()# in columns: the first number from right is the sh area code and the next number is slp class 
for x in week_day_dict.keys():
 tabaghe_bandi_matrixes_dict_avg[x] = cal_sum_average(week_day_dict[x], x, type_cal='tab')

"""#SSIM Formulation"""

def Lxy(ux , uy , C1=(10**-10)):
  return (2*ux*uy + C1)/ ( ux**2 + uy**2 + C1)

def Cxy(stdx, stdy, C2=(10**-2)):
  return (2*stdx*stdy + C2) / (stdx**2 + stdy**2 + C2)

def STR(cova ,stdx, stdy, C3=((10**-2)/2)): # we use pearson correlation coefficient instead of this formula
  return (cova + C3) / (stdx*stdy + C3)

def SSIM(Lxy, Cxy, STR, alpha=1, beta=1, Y=1):
  return (Lxy**alpha)*(Cxy**beta)*(STR**Y)

def df_to_SSIM(w_x , w_y):
  ''' this functin takes the 2 windows as a dataframe
   and outputs the SSIM coefficient and str'''
  #check the type if it's not numpy array turn it to numpy arrayr
  if type(w_x) is not np.ndarray:
    w_x = w_x.values
    w_y = w_y.values
  
  #flattern
  w_x = w_x.reshape(w_x.shape[0]*w_x.shape[1])
  w_y = w_y.reshape(w_y.shape[0]*w_y.shape[1])
  
  #calculate mean
  ux = np.mean(w_x)
  uy = np.mean(w_y)

  #calculate standarad deviation
  stdx = np.std(w_x)
  stdy = np.std(w_y)

  #insert the proccesed values to the main formula
  lxy = Lxy(ux , uy)
  cxy = Cxy(stdx, stdy)

  strxy = np.corrcoef(w_x, w_y)
  strxy = np.mean(strxy)

  #insert the proccesed values to the main formula
  output = SSIM(lxy ,cxy, strxy)
  return output , strxy

"""#Slide on the Matrix!"""

def slide(df, n_dim):
  '''gets the X or Y dataframe and returns a list of
  windows as a subset of the main dataframe'''
  list_df = list()#list of windows for each matrix
  for z in range(0, (len(df.index)+1)):
    for i in range(0, (len(df.columns)+1)):
      sub_df = df.iloc[z:(n_dim+z), i:(n_dim+i)]
      #only add the subsets with the same number of dimentions to the list of windows
      if len(sub_df.index) == n_dim and len(sub_df.columns) == n_dim:
        list_df.append(sub_df)
      else:
        pass
  return list_df

#the faster way than above it you can't get the details
def slide_np(df, n_dim):
  '''gets the X or Y dataframe and returns a list of
  windows as a subset of the main dataframe'''
  df_val = df.values
  list_df = list()#list of windows for each matrix
  for z in range(0, (len(df.index)+1)):
    for i in range(0, (len(df.columns)+1)):
      sub_df = df_val[z:(n_dim+z), i:(n_dim+i)]
      #only add the subsets with the same number of dimentions to the list of windows
      if sub_df.shape[1] == n_dim and sub_df.shape[0] == n_dim:
        list_df.append(sub_df)
      else:
        pass
  return list_df

def make_windows_pah_tab(df, input_list):
  list_windows = list()
  for x in input_list:
    df1 = df.filter(like=x)
    for z in input_list:
      list_index = [col for col in df.index if z in col]
      output = df1[df1.index.isin(list_index)]
      list_windows.append(output)
  return list_windows

# this will be used for taz and shahrdari areas
def cal_avrg_SSIM(window_list_x, window_list_y, mode='general'):
  '''this function computes all the ssim and outputs the average'''
  n=0
  ssim_list = list()
  strxy_list = list()

  if mode=='detail':
    df_detail = pd.DataFrame(columns=['windowx', 'windowy', 'SSIM'])

    for i in window_list_x:
      w1 = window_list_x[n]
      w2 = window_list_y[n]

      x = ','.join(str(e) for e in w1.columns)
      z = ','.join(str(e) for e in w1.index)

      #compare two windows in each loop and save the ssim
      ssim, strxy = df_to_SSIM(w1, w2)
      ssim_list.append(ssim)
      strxy_list.append(strxy)

      list_result = [x , z , strxy]

      series_strxy = pd.Series(list_result, index = df_detail.columns)
      df_detail = df_detail.append(series_strxy, ignore_index=True)
      n += 1

    strxy_detail =to_matrix(df_detail , 'windowx', 'windowy', 'SSIM')
    ssim_mean = np.mean(ssim_list)
    strxy_mean = np.mean(strxy_list)

    return ssim_mean, strxy_mean, strxy_detail
  else:#if we don't need calculation of detail we don't compute it
    for i in window_list_x:
      w1 = window_list_x[n]
      w2 = window_list_y[n]

      #compare two windows in each loop and save the ssim
      ssim, strxy = df_to_SSIM(w1, w2)
      ssim_list.append(ssim)
      strxy_list.append(strxy)

      n += 1

    ssim_mean = np.mean(ssim_list)
    strxy_mean = np.mean(strxy_list)

    return ssim_mean, strxy_mean

import os

os.makedirs('/content/shahrdari')
os.makedirs('/content/shahrdari/str')
os.makedirs('/content/shahrdari/ssim')

window_dim = [5, 10, 15, 25, 50, 75, 100, 117]
for b in window_dim:
  shahrdari_ssim_dict = dict()

  #add ssim's to this dataframe
  df_shahrdari_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
  df_shahrdari_strxy = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

  ndim = b
  for x in shahrdari_matrixes_dict_avg.keys():
    #get windows for compared matrices
    window_dict_x = slide_np(shahrdari_matrixes_dict_avg[x], ndim)
    for z in shahrdari_matrixes_dict_avg.keys():
      #get windows for compared matrices
      window_dict_y = slide_np(shahrdari_matrixes_dict_avg[z], ndim)
      result_ssim, result_strxy = cal_avrg_SSIM(window_dict_x, window_dict_y)

      list_result_ssim = [x , z , result_ssim]
      list_result_strxy = [x , z , result_strxy]

      a_series_ssim = pd.Series(list_result_ssim, index = df_shahrdari_ssim.columns)
      a_series_strxy = pd.Series(list_result_strxy, index = df_shahrdari_strxy.columns)

      df_shahrdari_ssim = df_shahrdari_ssim.append(a_series_ssim, ignore_index=True)
      df_shahrdari_strxy = df_shahrdari_strxy.append(a_series_strxy, ignore_index=True)
  df12=to_matrix(df_shahrdari_ssim , 'weekay_x', 'weekday_y', 'SSIM')
  df12.to_csv('/content/shahrdari/ssim/shahrdari_dim_'+str(ndim) +'_ssim_result.csv')

  df16=to_matrix(df_shahrdari_strxy , 'weekay_x', 'weekday_y', 'str')
  df16.to_csv('/content/shahrdari/str/shahrdari_dim_'+str(ndim) +'_str_result.csv')

"""##make windows for pahneh and tabaghe bandi """

os.makedirs('/content/pahneh')
os.makedirs('/content/pahneh/detail')
os.makedirs('/content/pahneh/str')
os.makedirs('/content/pahneh/ssim')

list_pahneh = ['Center', 'South', 'East', 'West', 'North']

#add ssim's to this dataframe
df_pahneh_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
df_pahneh_str = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

for x in pahneh_slp_matrixes_dict_avg.keys():
  #get windows for compared matrices
  window_dict_x = make_windows_pah_tab(pahneh_slp_matrixes_dict_avg[x], list_pahneh)
  for z in pahneh_slp_matrixes_dict_avg.keys():
    #get windows for compared matrices
    window_dict_y = make_windows_pah_tab(pahneh_slp_matrixes_dict_avg[z], list_pahneh)
    result_ssim, result_str, detail = cal_avrg_SSIM(window_dict_x, window_dict_y, mode='detail')

    detail.to_csv('/content/pahneh/detail/detailed_ssim'+'_'+x+'_'+z+'.csv')

    list_result_ssim = [x , z , result_ssim]
    a_series_ssim = pd.Series(list_result_ssim, index = df_pahneh_ssim.columns)

    list_result_str = [x , z , result_str]
    a_series_str = pd.Series(list_result_str, index = df_pahneh_str.columns)

    df_pahneh_ssim = df_pahneh_ssim.append(a_series_ssim, ignore_index=True)
    df_pahneh_str = df_pahneh_str.append(a_series_str, ignore_index=True)

df13=to_matrix(df_pahneh_ssim , 'weekay_x', 'weekday_y', 'SSIM')
df13.to_csv('/content/pahneh/ssim/pahneh_ssim_result.csv')

df17=to_matrix(df_pahneh_str , 'weekay_x', 'weekday_y', 'str')
df17.to_csv('/content/pahneh/str/pahneh_str_result.csv')

slp_pahneh = list()
slp_binary = ['0','1']
for i in list_pahneh:
  for z in slp_binary:
    slp_element = i + '_' + z
    slp_pahneh.append(slp_element)

os.makedirs('/content/pahneh_slp')
os.makedirs('/content/pahneh_slp/detail')
os.makedirs('/content/pahneh_slp/str')
os.makedirs('/content/pahneh_slp/ssim')

#add ssim's to this dataframe
df_pahneh_slp_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
df_pahneh_slp_str = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

for x in pahneh_slp_matrixes_dict_avg.keys():
  #get windows for compared matrices
  window_dict_x = make_windows_pah_tab(pahneh_slp_matrixes_dict_avg[x], slp_pahneh)
  for z in pahneh_slp_matrixes_dict_avg.keys():
    #get windows for compared matrices
    window_dict_y = make_windows_pah_tab(pahneh_slp_matrixes_dict_avg[z], slp_pahneh)
    result_ssim, result_str, detail = cal_avrg_SSIM(window_dict_x, window_dict_y, mode='detail')

    detail.to_csv('/content/pahneh_slp/detail/detailed_ssim'+'_'+x+'_'+z+'.csv')

    list_result_ssim = [x , z , result_ssim]
    a_series_ssim = pd.Series(list_result_ssim, index = df_pahneh_slp_ssim.columns)

    list_result_str = [x , z , result_str]
    a_series_str = pd.Series(list_result_str, index = df_pahneh_slp_str.columns)

    df_pahneh_slp_ssim = df_pahneh_slp_ssim.append(a_series_ssim, ignore_index=True)
    df_pahneh_slp_str = df_pahneh_slp_str.append(a_series_str, ignore_index=True)

df14=to_matrix(df_pahneh_slp_ssim , 'weekay_x', 'weekday_y', 'SSIM')
df14.to_csv('/content/pahneh_slp/ssim/pahneh_slp_ssim_result.csv')

df18=to_matrix(df_pahneh_slp_str , 'weekay_x', 'weekday_y', 'str')
df18.to_csv('/content/pahneh_slp/str/pahneh_slp_str_result.csv')

os.makedirs('/content/tabaghe_bandi')
os.makedirs('/content/tabaghe_bandi/detail')
os.makedirs('/content/tabaghe_bandi/str')
os.makedirs('/content/tabaghe_bandi/ssim')

#make needed lists for the below function
tab_list = ['cluster_1','cluster_2','cluster_3','cluster_4','cluster_5']

#add ssim's to this dataframe
df_tab_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
df_tab_str = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])
for x in tabaghe_bandi_matrixes_dict_avg.keys():
  #get windows for compared matrices
  window_dict_x = make_windows_pah_tab(tabaghe_bandi_matrixes_dict_avg[x], tab_list)
  for z in tabaghe_bandi_matrixes_dict_avg.keys():
    #get windows for compared matrices
    window_dict_y = make_windows_pah_tab(tabaghe_bandi_matrixes_dict_avg[z], tab_list)
    result_ssim, result_str, detail = cal_avrg_SSIM(window_dict_x, window_dict_y, mode='detail')

    detail.to_csv('/content/tabaghe_bandi/detail/detailed_ssim'+'_'+x+'_'+z+'.csv')

    list_result_ssim = [x , z , result_ssim]
    a_series_ssim = pd.Series(list_result_ssim, index = df_tab_ssim.columns)

    list_result_str = [x , z , result_str]
    a_series_str = pd.Series(list_result_str, index = df_tab_str.columns)

    df_tab_ssim = df_tab_ssim.append(a_series_ssim, ignore_index=True)
    df_tab_str = df_tab_str.append(a_series_str, ignore_index=True)

df15=to_matrix(df_tab_ssim , 'weekay_x', 'weekday_y', 'SSIM')
df15.to_csv('/content/tabaghe_bandi/ssim/tabaghe_bandi_ssim_result.csv')

df19=to_matrix(df_tab_str , 'weekay_x', 'weekday_y', 'str')
df19.to_csv('/content/tabaghe_bandi/str/tabaghe_bandi_str_result.csv')

!zip -r /content/taz.zip /content/taz
from google.colab import files
files.download("/content/taz.zip")

!zip -r /content/shahrdari.zip /content/shahrdari
from google.colab import files
files.download("/content/shahrdari.zip")

from google.colab import files
!zip -r /content/pahneh_slp.zip /content/pahneh_slp
files.download("/content/pahneh_slp.zip")

!zip -r /content/pahneh.zip /content/pahneh
files.download("/content/pahneh.zip")

!zip -r /content/tabaghe_bandi.zip /content/tabaghe_bandi
files.download("/content/tabaghe_bandi.zip")

"""#motaleate jameh comparison"""

#taz_mid_week_avrg = (taz_matrixes_dict_avg['Sunday'] + taz_matrixes_dict_avg['Monday']+ taz_matrixes_dict_avg['Tuesday'])/3
shahrdari_mid_week_avrg = (shahrdari_matrixes_dict_avg['Sunday'] + shahrdari_matrixes_dict_avg['Monday'] + shahrdari_matrixes_dict_avg['Tuesday'])/3
pahneh_slp_mid_week_avrg = (pahneh_slp_matrixes_dict_avg['Sunday'] + pahneh_slp_matrixes_dict_avg['Monday']+ pahneh_slp_matrixes_dict_avg['Tuesday'])/3
tabaghe_bandi_mid_week_avrg = (tabaghe_bandi_matrixes_dict_avg['Sunday'] + tabaghe_bandi_matrixes_dict_avg['Monday']+ tabaghe_bandi_matrixes_dict_avg['Tuesday'])/3

taz_mid_week_avrg_10 = taz_mid_week_avrg/10
shahrdari_mid_week_avrg_10 =shahrdari_mid_week_avrg/10
pahneh_slp_mid_week_avrg_10 =pahneh_slp_mid_week_avrg/10
tabaghe_bandi_mid_week_avrg_10 =tabaghe_bandi_mid_week_avrg/10

pb_df_hour = pd.read_csv('/content/drive/MyDrive/pb_df_hour.csv', index_col=0)
pv_df_hour = pd.read_csv('/content/drive/MyDrive/pv_df_hour.csv', index_col=0)

pb_df_hour = pb_df_hour[pb_df_hour['Origin'] <= 700]
pb_df_hour = pb_df_hour[pb_df_hour['Destination'] <= 700]

pv_df_hour = pv_df_hour[pv_df_hour['Destination'] <= 700]
pv_df_hour = pv_df_hour[pv_df_hour['Origin'] <= 700]

pb_df_hour['sum'] = pb_df_hour.iloc[: , 2:].sum(axis=1)
pv_df_hour['sum'] = pv_df_hour.iloc[: , 2:].sum(axis=1)

pb_taz_sum = pb_df_hour[['Origin', 'Destination', 'sum']]
pv_taz_sum = pv_df_hour[['Origin', 'Destination', 'sum']]

#taz
pb_taz_sum_matrix = to_matrix(pb_taz_sum , 'Origin', 'Destination', 'sum')
pv_taz_sum_matrix = to_matrix(pv_taz_sum , 'Origin', 'Destination', 'sum')

pb_taz_pk = pb_df_hour[['Origin', 'Destination', 'q']]
pv_taz_pk = pv_df_hour[['Origin', 'Destination', 'q']]

#taz
pb_taz_pk_matrix = to_matrix(pb_taz_pk , 'Origin', 'Destination', 'q')
pv_taz_pk_matrix = to_matrix(pv_taz_pk , 'Origin', 'Destination', 'q')

pb_df_hour['sh_o_agg'] = pb_df_hour['Origin']
pb_df_hour['sh_d_agg'] = pb_df_hour['Destination']

#pb_df_hour = pb_df_hour[pb_df_hour['sh_o_agg'] != 51]
#pb_df_hour = pb_df_hour[pb_df_hour['sh_d_agg'] != 51]
pb_sh = pb_df_hour.dropna()

pb_sh.reset_index(drop=True, inplace=True)
pb_sh_sum = pb_sh.groupby(['sh_o_agg','sh_d_agg'])['sum'].agg(['sum'])
pb_sh_sum = pb_sh_sum.reset_index()
pb_sh_sum_matrix = to_matrix(pb_sh_sum, 'sh_o_agg', 'sh_d_agg', 'sum')

pb_sh.reset_index(drop=True, inplace=True)
pb_sh_pk = pb_sh.groupby(['sh_o_agg','sh_d_agg'])['q'].agg(['sum'])
pb_sh_pkr = pb_sh_pk.reset_index()
pb_sh_pk_matrix = to_matrix(pb_sh_pk, 'sh_o_agg', 'sh_d_agg', 'sum')

####pv
pv_df_hour['sh_o_agg'] = pv_df_hour['Origin']
pv_df_hour['sh_d_agg'] = pv_df_hour['Destination']

#pv_df_hour = pv_df_hour[pv_df_hour['sh_o_agg'] != 51]
#pv_df_hour = pv_df_hour[pv_df_hour['sh_d_agg'] != 51]
                        
pv_sh = pv_df_hour.dropna()

pv_sh.reset_index(drop=True, inplace=True)
pv_sh_sum = pv_sh.groupby(['sh_o_agg','sh_d_agg'])['sum'].agg(['sum'])
pv_sh_sum = pv_sh_sum.reset_index()
pv_sh_sum_matrix = to_matrix(pv_sh_sum, 'sh_o_agg', 'sh_d_agg', 'sum')

pv_sh.reset_index(drop=True, inplace=True)
pv_sh_pk = pv_sh.groupby(['sh_o_agg','sh_d_agg'])['q'].agg(['sum'])
pv_sh_pkr = pv_sh_pk.reset_index()
pv_sh_pk_matrix = to_matrix(pv_sh_pk, 'sh_o_agg', 'sh_d_agg', 'sum')

# shahrdari area to pahneh
pv_df_hour['pah_o_agg'] = pv_df_hour['sh_o_agg'].map(pahneh_dict)
pv_df_hour['pah_d_agg'] = pv_df_hour['sh_d_agg'].map(pahneh_dict)

# slp classification
pv_df_hour['slp_o_agg'] = pv_df_hour['sh_o_agg'].map(slp_dict)
pv_df_hour['slp_d_agg'] = pv_df_hour['sh_d_agg'].map(slp_dict)

pv_df_hour.dropna(inplace=True)
pv_df_hour['pah_sh_slp_o_agg'] = pv_df_hour['pah_o_agg'].astype('str') +'_'+ (pv_df_hour['slp_o_agg'].astype('int32')).astype('str') \
                               +'_'+ (pv_df_hour['sh_o_agg'].astype('int32')).astype('str')
pv_df_hour['pah_sh_slp_d_agg'] = pv_df_hour['pah_d_agg'].astype('str') +'_'+ (pv_df_hour['slp_d_agg'].astype('int32')).astype('str') \
                               +'_'+ (pv_df_hour['sh_d_agg'].astype('int32')).astype('str')

# needed to aggregate over simillar areacode combinations to prevent huge loss of trips
pv_pah_slp = pv_df_hour.reset_index(drop=True)
pv_pah_slp_sum = pv_pah_slp.groupby(['pah_sh_slp_o_agg','pah_sh_slp_d_agg'])['sum'].agg(['sum'])
pv_pah_slp_sum = pv_pah_slp_sum.rename(columns={'sum':'Count'}).reset_index()
pv_pah_slp_sum_matrix = to_matrix(pv_pah_slp_sum, 'pah_sh_slp_o_agg', 'pah_sh_slp_d_agg', 'Count')

# needed to aggregate over simillar areacode combinations to prevent huge loss of trips
pv_pah_slp_pk = pv_pah_slp.groupby(['pah_sh_slp_o_agg','pah_sh_slp_d_agg'])['q'].agg(['sum'])
pv_pah_slp_pk = pv_pah_slp_pk.rename(columns={'sum':'Count'}).reset_index()
pv_pah_slp_matrix_pk = to_matrix(pv_pah_slp_pk, 'pah_sh_slp_o_agg', 'pah_sh_slp_d_agg', 'Count')


###pbb
# shahrdari area to pahneh
pb_df_hour['pah_o_agg'] = pb_df_hour['sh_o_agg'].map(pahneh_dict)
pb_df_hour['pah_d_agg'] = pb_df_hour['sh_d_agg'].map(pahneh_dict)

# slp classification
pb_df_hour['slp_o_agg'] = pb_df_hour['sh_o_agg'].map(slp_dict)
pb_df_hour['slp_d_agg'] = pb_df_hour['sh_d_agg'].map(slp_dict)
pb_df_hour.dropna(inplace=True)

pb_df_hour['pah_sh_slp_o_agg'] = pb_df_hour['pah_o_agg'].astype('str') +'_'+ (pb_df_hour['slp_o_agg'].astype('int32')).astype('str') \
                               +'_'+ (pb_df_hour['sh_o_agg'].astype('int32')).astype('str')

pb_df_hour['pah_sh_slp_d_agg'] = pb_df_hour['pah_d_agg'].astype('str') +'_'+ (pb_df_hour['slp_d_agg'].astype('int32')).astype('str') \
                               +'_'+ (pb_df_hour['sh_d_agg'].astype('int32')).astype('str')

# needed to aggregate over simillar areacode combinations to prevent huge loss of trips
pb_pah_slp = pv_df_hour.reset_index(drop=True)
pb_pah_slp_sum = pb_pah_slp.groupby(['pah_sh_slp_o_agg','pah_sh_slp_d_agg'])['sum'].agg(['sum'])
pb_pah_slp_sum = pb_pah_slp_sum.rename(columns={'sum':'Count'}).reset_index()
pb_pah_slp_sum_matrix = to_matrix(pb_pah_slp_sum, 'pah_sh_slp_o_agg', 'pah_sh_slp_d_agg', 'Count')

# needed to aggregate over simillar areacode combinations to prevent huge loss of trips
pb_pah_slp_pk = pb_pah_slp.groupby(['pah_sh_slp_o_agg','pah_sh_slp_d_agg'])['q'].agg(['sum'])
pb_pah_slp_pk = pb_pah_slp_pk.rename(columns={'sum':'Count'}).reset_index()
pb_pah_slp_matrix_pk = to_matrix(pb_pah_slp_pk, 'pah_sh_slp_o_agg', 'pah_sh_slp_d_agg', 'Count')

def df_to_pah(df):
  df['tab_o_agg'] = df['sh_o_agg'].map(tabaghe_bandi_dict)
  df['tab_d_agg'] = df['sh_d_agg'].map(tabaghe_bandi_dict)
      
  df['tab_sh_o_agg'] = df['tab_o_agg'].astype('str') +'_'+ (df['sh_o_agg'].astype('int32')).astype('str') 

  df['tab_sh_d_agg'] = df['tab_d_agg'].astype('str') +'_'+ (df['sh_d_agg'].astype('int32')).astype('str')

  # needed to aggregate over simillar areacode combinations to prevent huge loss of trips
  df.reset_index(drop=True, inplace=True)
  df1 = df.groupby(['tab_sh_o_agg','tab_sh_d_agg'])['sum'].agg(['sum'])
  df1 = df1.rename(columns={'sum':'Count'}).reset_index()
  OD_matrix_1 = to_matrix(df1, 'tab_sh_o_agg', 'tab_sh_d_agg', 'Count')

  # needed to aggregate over simillar areacode combinations to prevent huge loss of trips
  df2 = df.groupby(['tab_sh_o_agg','tab_sh_d_agg'])['q'].agg(['sum'])
  df2 = df2.rename(columns={'sum':'Count'}).reset_index()
  OD_matrix_2 = to_matrix(df2, 'tab_sh_o_agg', 'tab_sh_d_agg', 'Count')
  return OD_matrix_1, OD_matrix_2

pb_tab_slp_matrix_sum , pb_tab_slp_matrix_pk = df_to_pah(pb_df_hour)

pv_tab_slp_matrix_sum , pv_tab_slp_matrix_pk = df_to_pah(pv_df_hour)

###fin

taz_sum_matrix = pb_taz_sum_matrix + pv_taz_sum_matrix
taz_pk_matrix = pb_taz_pk_matrix + pv_taz_pk_matrix

#sum shahradi trips
sh_sum_matrix = pv_sh_sum_matrix + pb_sh_sum_matrix

sh_pk_matrix = pv_sh_pk_matrix + pb_sh_pk_matrix

pah_slp_pk_matrix = pb_pah_slp_matrix_pk + pv_pah_slp_matrix_pk

pah_slp_sum_matrix = pb_pah_slp_sum_matrix + pv_pah_slp_sum_matrix

tab_matrix_sum = pb_tab_slp_matrix_sum + pv_tab_slp_matrix_sum

tab_matrix_pk = pb_tab_slp_matrix_pk + pv_tab_slp_matrix_pk

"""#Calculate ssim"""

taz_mid_week_avrg_10 = taz_mid_week_avrg/10
shahrdari_mid_week_avrg_10 =shahrdari_mid_week_avrg/10
pahneh_slp_mid_week_avrg_10 =pahneh_slp_mid_week_avrg/10
tabaghe_bandi_mid_week_avrg_10 =tabaghe_bandi_mid_week_avrg/10

"""##peak hour"""

import os

os.makedirs('/content/taz')
os.makedirs('/content/taz/str')
os.makedirs('/content/taz/ssim')

window_dim = [25, 50, 75, 100, 700]
for b in window_dim:
  shahrdari_ssim_dict = dict()

  #add ssim's to this dataframe
  df_shahrdari_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
  df_shahrdari_strxy = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

  ndim = b
  #get windows for compared matrices
  window_dict_x = slide_np(taz_mid_week_avrg_10, ndim)

  #get windows for compared matrices
  window_dict_y = slide_np(taz_pk_matrix, ndim)
  result_ssim, result_strxy = cal_avrg_SSIM(window_dict_x, window_dict_y)

  list_result_ssim = ['x' , 'z' , result_ssim]
  list_result_strxy = ['x' , 'z' , result_strxy]

  a_series_ssim = pd.Series(list_result_ssim, index = df_shahrdari_ssim.columns)
  a_series_strxy = pd.Series(list_result_strxy, index = df_shahrdari_strxy.columns)

  df_shahrdari_ssim = df_shahrdari_ssim.append(a_series_ssim, ignore_index=True)
  df_shahrdari_strxy = df_shahrdari_strxy.append(a_series_strxy, ignore_index=True)

  df12=to_matrix(df_shahrdari_ssim , 'weekay_x', 'weekday_y', 'SSIM')
  df12.to_csv('/content/taz/ssim/taz_dim_'+str(ndim) +'_ssim_result.csv')

  df16=to_matrix(df_shahrdari_strxy , 'weekay_x', 'weekday_y', 'str')
  df16.to_csv('/content/taz/str/taz_dim_'+str(ndim) +'_str_result.csv')

import os

os.makedirs('/content/shahrdari')
os.makedirs('/content/shahrdari/str')
os.makedirs('/content/shahrdari/ssim')

window_dim = [5, 10, 15, 25, 50, 75, 100, 117]
for b in window_dim:
  shahrdari_ssim_dict = dict()

  #add ssim's to this dataframe
  df_shahrdari_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
  df_shahrdari_strxy = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

  ndim = b
  #get windows for compared matrices
  window_dict_x = slide_np(shahrdari_mid_week_avrg_10, ndim)

  #get windows for compared matrices
  window_dict_y = slide_np(sh_pk_matrix, ndim)
  result_ssim, result_strxy = cal_avrg_SSIM(window_dict_x, window_dict_y)

  list_result_ssim = ['x' , 'y' , result_ssim]
  list_result_strxy = ['x' , 'y' , result_strxy]

  a_series_ssim = pd.Series(list_result_ssim, index = df_shahrdari_ssim.columns)
  a_series_strxy = pd.Series(list_result_strxy, index = df_shahrdari_strxy.columns)

  df_shahrdari_ssim = df_shahrdari_ssim.append(a_series_ssim, ignore_index=True)
  df_shahrdari_strxy = df_shahrdari_strxy.append(a_series_strxy, ignore_index=True)

  df12=to_matrix(df_shahrdari_ssim , 'weekay_x', 'weekday_y', 'SSIM')
  df12.to_csv('/content/shahrdari/ssim/shahrdari_dim_'+str(ndim) +'_ssim_result.csv')

  df16=to_matrix(df_shahrdari_strxy , 'weekay_x', 'weekday_y', 'str')
  df16.to_csv('/content/shahrdari/str/shahrdari_dim_'+str(ndim) +'_str_result.csv')

os.makedirs('/content/pahneh')
os.makedirs('/content/pahneh/detail')
os.makedirs('/content/pahneh/str')
os.makedirs('/content/pahneh/ssim')

list_pahneh = ['Center', 'South', 'East', 'West', 'North']

#add ssim's to this dataframe
df_pahneh_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
df_pahneh_str = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

window_dict_x = make_windows_pah_tab(pahneh_slp_mid_week_avrg_10, list_pahneh)
#get windows for compared matrices
window_dict_y = make_windows_pah_tab(pah_slp_pk_matrix, list_pahneh)
result_ssim, result_str, detail = cal_avrg_SSIM(window_dict_x, window_dict_y, mode='detail')

detail.to_csv('/content/pahneh/detail/detailed_str'+'.csv')
list_result_ssim = ['x ', 'z' , result_ssim]
a_series_ssim = pd.Series(list_result_ssim, index = df_pahneh_ssim.columns)

list_result_str = ['x' , 'z' , result_str]
a_series_str = pd.Series(list_result_str, index = df_pahneh_str.columns)

df_pahneh_ssim = df_pahneh_ssim.append(a_series_ssim, ignore_index=True)
df_pahneh_str = df_pahneh_str.append(a_series_str, ignore_index=True)

df13=to_matrix(df_pahneh_ssim , 'weekay_x', 'weekday_y', 'SSIM')
df13.to_csv('/content/pahneh/ssim/pahneh_ssim_result.csv')

df17=to_matrix(df_pahneh_str , 'weekay_x', 'weekday_y', 'str')
df17.to_csv('/content/pahneh/str/pahneh_str_result.csv')

slp_pahneh = list()
slp_binary = ['0','1']
for i in list_pahneh:
  for z in slp_binary:
    slp_element = i + '_' + z
    slp_pahneh.append(slp_element)

os.makedirs('/content/pahneh_slp')
os.makedirs('/content/pahneh_slp/detail')
os.makedirs('/content/pahneh_slp/str')
os.makedirs('/content/pahneh_slp/ssim')

#add ssim's to this dataframe
df_pahneh_slp_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
df_pahneh_slp_str = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

window_dict_x = make_windows_pah_tab(pahneh_slp_mid_week_avrg_10, slp_pahneh)
#get windows for compared matrices
window_dict_y = make_windows_pah_tab(pah_slp_pk_matrix, slp_pahneh)
result_ssim, result_str, detail = cal_avrg_SSIM(window_dict_x, window_dict_y, mode='detail')

detail.to_csv('/content/pahneh_slp/detail/detailed_str'+'.csv')

list_result_ssim = ['x' , 'z' , result_ssim]
a_series_ssim = pd.Series(list_result_ssim, index = df_pahneh_slp_ssim.columns)

list_result_str = ['x' , 'z' , result_str]
a_series_str = pd.Series(list_result_str, index = df_pahneh_slp_str.columns)

df_pahneh_slp_ssim = df_pahneh_slp_ssim.append(a_series_ssim, ignore_index=True)
df_pahneh_slp_str = df_pahneh_slp_str.append(a_series_str, ignore_index=True)

df14=to_matrix(df_pahneh_slp_ssim , 'weekay_x', 'weekday_y', 'SSIM')
df14.to_csv('/content/pahneh_slp/ssim/pahneh_slp_ssim_result.csv')

df18=to_matrix(df_pahneh_slp_str , 'weekay_x', 'weekday_y', 'str')
df18.to_csv('/content/pahneh_slp/str/pahneh_slp_str_result.csv')

os.makedirs('/content/tabaghe_bandi')
os.makedirs('/content/tabaghe_bandi/detail')
os.makedirs('/content/tabaghe_bandi/str')
os.makedirs('/content/tabaghe_bandi/ssim')

#make needed lists for the below function
tab_list = ['cluster_1','cluster_2','cluster_3','cluster_4','cluster_5']

#add ssim's to this dataframe
df_tab_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
df_tab_str = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

#get windows for compared matrices
window_dict_x = make_windows_pah_tab(tabaghe_bandi_mid_week_avrg_10, tab_list)

#get windows for compared matrices
window_dict_y = make_windows_pah_tab(tab_matrix_pk, tab_list)
result_ssim, result_str, detail = cal_avrg_SSIM(window_dict_x, window_dict_y, mode='detail')

detail.to_csv('/content/tabaghe_bandi/detail/detailed_str'+'.csv')

list_result_ssim = ['x' , 'z' , result_ssim]
a_series_ssim = pd.Series(list_result_ssim, index = df_tab_ssim.columns)

list_result_str = ['x' , 'z' , result_str]
a_series_str = pd.Series(list_result_str, index = df_tab_str.columns)

df_tab_ssim = df_tab_ssim.append(a_series_ssim, ignore_index=True)
df_tab_str = df_tab_str.append(a_series_str, ignore_index=True)

df15=to_matrix(df_tab_ssim , 'weekay_x', 'weekday_y', 'SSIM')
df15.to_csv('/content/tabaghe_bandi/ssim/tabaghe_bandi_ssim_result.csv')

df19=to_matrix(df_tab_str , 'weekay_x', 'weekday_y', 'str')
df19.to_csv('/content/tabaghe_bandi/str/tabaghe_bandi_str_result.csv')

"""##daily"""

import os

os.makedirs('/content/taz')
os.makedirs('/content/taz/str')
os.makedirs('/content/taz/ssim')

window_dim = [25, 50, 75, 100, 700]
for b in window_dim:
  shahrdari_ssim_dict = dict()

  #add ssim's to this dataframe
  df_shahrdari_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
  df_shahrdari_strxy = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

  ndim = b
  #get windows for compared matrices
  window_dict_x = slide_np(taz_mid_week_avrg, ndim)

  #get windows for compared matrices
  window_dict_y = slide_np(taz_sum_matrix, ndim)
  result_ssim, result_strxy = cal_avrg_SSIM(window_dict_x, window_dict_y)

  list_result_ssim = ['x' , 'z' , result_ssim]
  list_result_strxy = ['x' , 'z' , result_strxy]

  a_series_ssim = pd.Series(list_result_ssim, index = df_shahrdari_ssim.columns)
  a_series_strxy = pd.Series(list_result_strxy, index = df_shahrdari_strxy.columns)

  df_shahrdari_ssim = df_shahrdari_ssim.append(a_series_ssim, ignore_index=True)
  df_shahrdari_strxy = df_shahrdari_strxy.append(a_series_strxy, ignore_index=True)

  df12=to_matrix(df_shahrdari_ssim , 'weekay_x', 'weekday_y', 'SSIM')
  df12.to_csv('/content/taz/ssim/taz_dim_'+str(ndim) +'_ssim_result.csv')

  df16=to_matrix(df_shahrdari_strxy , 'weekay_x', 'weekday_y', 'str')
  df16.to_csv('/content/taz/str/taz_dim_'+str(ndim) +'_str_result.csv')

import os

os.makedirs('/content/shahrdari')
os.makedirs('/content/shahrdari/str')
os.makedirs('/content/shahrdari/ssim')

window_dim = [5, 10, 15, 25, 50, 75, 100, 117]
for b in window_dim:
  shahrdari_ssim_dict = dict()

  #add ssim's to this dataframe
  df_shahrdari_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
  df_shahrdari_strxy = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

  ndim = b
  #get windows for compared matrices
  window_dict_x = slide_np(shahrdari_mid_week_avrg, ndim)

  #get windows for compared matrices
  window_dict_y = slide_np(sh_sum_matrix, ndim)
  result_ssim, result_strxy = cal_avrg_SSIM(window_dict_x, window_dict_y)

  list_result_ssim = ['x' , 'y' , result_ssim]
  list_result_strxy = ['x' , 'y' , result_strxy]

  a_series_ssim = pd.Series(list_result_ssim, index = df_shahrdari_ssim.columns)
  a_series_strxy = pd.Series(list_result_strxy, index = df_shahrdari_strxy.columns)

  df_shahrdari_ssim = df_shahrdari_ssim.append(a_series_ssim, ignore_index=True)
  df_shahrdari_strxy = df_shahrdari_strxy.append(a_series_strxy, ignore_index=True)

  df12=to_matrix(df_shahrdari_ssim , 'weekay_x', 'weekday_y', 'SSIM')
  df12.to_csv('/content/shahrdari/ssim/shahrdari_dim_'+str(ndim) +'_ssim_result.csv')

  df16=to_matrix(df_shahrdari_strxy , 'weekay_x', 'weekday_y', 'str')
  df16.to_csv('/content/shahrdari/str/shahrdari_dim_'+str(ndim) +'_str_result.csv')

os.makedirs('/content/pahneh')
os.makedirs('/content/pahneh/detail')
os.makedirs('/content/pahneh/str')
os.makedirs('/content/pahneh/ssim')

list_pahneh = ['Center', 'South', 'East', 'West', 'North']

#add ssim's to this dataframe
df_pahneh_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
df_pahneh_str = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

window_dict_x = make_windows_pah_tab(pahneh_slp_mid_week_avrg, list_pahneh)
#get windows for compared matrices
window_dict_y = make_windows_pah_tab(pah_slp_sum_matrix, list_pahneh)
result_ssim, result_str, detail = cal_avrg_SSIM(window_dict_x, window_dict_y, mode='detail')

detail.to_csv('/content/pahneh/detail/detailed_str'+'.csv')
list_result_ssim = ['x ', 'z' , result_ssim]
a_series_ssim = pd.Series(list_result_ssim, index = df_pahneh_ssim.columns)

list_result_str = ['x' , 'z' , result_str]
a_series_str = pd.Series(list_result_str, index = df_pahneh_str.columns)

df_pahneh_ssim = df_pahneh_ssim.append(a_series_ssim, ignore_index=True)
df_pahneh_str = df_pahneh_str.append(a_series_str, ignore_index=True)

df13=to_matrix(df_pahneh_ssim , 'weekay_x', 'weekday_y', 'SSIM')
df13.to_csv('/content/pahneh/ssim/pahneh_ssim_result.csv')

df17=to_matrix(df_pahneh_str , 'weekay_x', 'weekday_y', 'str')
df17.to_csv('/content/pahneh/str/pahneh_str_result.csv')

slp_pahneh = list()
slp_binary = ['0','1']
for i in list_pahneh:
  for z in slp_binary:
    slp_element = i + '_' + z
    slp_pahneh.append(slp_element)

os.makedirs('/content/pahneh_slp')
os.makedirs('/content/pahneh_slp/detail')
os.makedirs('/content/pahneh_slp/str')
os.makedirs('/content/pahneh_slp/ssim')

#add ssim's to this dataframe
df_pahneh_slp_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
df_pahneh_slp_str = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

window_dict_x = make_windows_pah_tab(pahneh_slp_mid_week_avrg, slp_pahneh)
#get windows for compared matrices
window_dict_y = make_windows_pah_tab(pah_slp_sum_matrix, slp_pahneh)
result_ssim, result_str, detail = cal_avrg_SSIM(window_dict_x, window_dict_y, mode='detail')

detail.to_csv('/content/pahneh_slp/detail/detailed_str'+'.csv')

list_result_ssim = ['x' , 'z' , result_ssim]
a_series_ssim = pd.Series(list_result_ssim, index = df_pahneh_slp_ssim.columns)

list_result_str = ['x' , 'z' , result_str]
a_series_str = pd.Series(list_result_str, index = df_pahneh_slp_str.columns)

df_pahneh_slp_ssim = df_pahneh_slp_ssim.append(a_series_ssim, ignore_index=True)
df_pahneh_slp_str = df_pahneh_slp_str.append(a_series_str, ignore_index=True)

df14=to_matrix(df_pahneh_slp_ssim , 'weekay_x', 'weekday_y', 'SSIM')
df14.to_csv('/content/pahneh_slp/ssim/pahneh_slp_ssim_result.csv')

df18=to_matrix(df_pahneh_slp_str , 'weekay_x', 'weekday_y', 'str')
df18.to_csv('/content/pahneh_slp/str/pahneh_slp_str_result.csv')

os.makedirs('/content/tabaghe_bandi')
os.makedirs('/content/tabaghe_bandi/detail')
os.makedirs('/content/tabaghe_bandi/str')
os.makedirs('/content/tabaghe_bandi/ssim')

#make needed lists for the below function
tab_list = ['cluster_1','cluster_2','cluster_3','cluster_4','cluster_5']

#add ssim's to this dataframe
df_tab_ssim = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'SSIM'])
df_tab_str = pd.DataFrame(columns=['weekay_x', 'weekday_y', 'str'])

#get windows for compared matrices
window_dict_x = make_windows_pah_tab(tabaghe_bandi_mid_week_avrg, tab_list)

#get windows for compared matrices
window_dict_y = make_windows_pah_tab(tab_matrix_sum, tab_list)
result_ssim, result_str, detail = cal_avrg_SSIM(window_dict_x, window_dict_y, mode='detail')

detail.to_csv('/content/tabaghe_bandi/detail/detailed_str'+'.csv')

list_result_ssim = ['x' , 'z' , result_ssim]
a_series_ssim = pd.Series(list_result_ssim, index = df_tab_ssim.columns)

list_result_str = ['x' , 'z' , result_str]
a_series_str = pd.Series(list_result_str, index = df_tab_str.columns)

df_tab_ssim = df_tab_ssim.append(a_series_ssim, ignore_index=True)
df_tab_str = df_tab_str.append(a_series_str, ignore_index=True)

df15=to_matrix(df_tab_ssim , 'weekay_x', 'weekday_y', 'SSIM')
df15.to_csv('/content/tabaghe_bandi/ssim/tabaghe_bandi_ssim_result.csv')

df19=to_matrix(df_tab_str , 'weekay_x', 'weekday_y', 'str')
df19.to_csv('/content/tabaghe_bandi/str/tabaghe_bandi_str_result.csv')
