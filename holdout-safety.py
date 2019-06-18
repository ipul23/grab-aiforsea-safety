import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from numpy.fft import *
from sklearn.model_selection import cross_val_score,train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.style as style 
import eli5
from skopt import BayesSearchCV
from eli5.sklearn import PermutationImportance
style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.

def preprocessing(data):
    PI = 3.1415926535897932384626433832795
    #sort data by `bookingID` and `second`
    data = data.sort_values(by=['bookingID','second'])
    #Convert acceleration unit to gravitational unit (1g = 9.81 m/s^2)
    data['acceleration_x'] = data['acceleration_x']/9.81
    data['acceleration_y'] = data['acceleration_y']/9.81
    data['acceleration_z'] = data['acceleration_z']/9.81
    #Extract `roll`, `pitch` and `tilt_angle`
    data['roll'] = np.arctan2(data['acceleration_y'] , data['acceleration_z'])*180.0 / PI
    data['pitch'] = np.arctan2((-data['acceleration_x']),np.sqrt(data['acceleration_y']**2 + data['acceleration_z']**2) )*180.0 / PI
    data['tilt_angle'] = np.arccos(data['acceleration_z']  / np.sqrt(data['acceleration_x']**2 + data['acceleration_y']**2 + data['acceleration_z']**2))*180/PI
    data['acc_tot'] = np.sqrt(data['acceleration_x']**2+data['acceleration_y']**2+data['acceleration_z']**2)
    data['gyro_tot'] = np.sqrt(data['gyro_x']**2+data['gyro_y']**2+data['gyro_z']**2)
    #Binning the `Bearing` feature
    data['bearing_bin'] = ''
    idx = data[data['Bearing'] <= 90]['bearing_bin'].index
    data['bearing_bin'].loc[idx] = 'NE'
    idx = data[(data['Bearing'] > 90) & (data['Bearing'] <= 180)]['bearing_bin'].index
    data['bearing_bin'].loc[idx] = 'ES'
    idx = data[(data['Bearing'] > 180) & (data['Bearing'] <= 270)]['bearing_bin'].index
    data['bearing_bin'].loc[idx] = 'SW'
    idx = data[(data['Bearing'] > 270) & (data['Bearing'] <= 360)]['bearing_bin'].index
    data['bearing_bin'].loc[idx] = 'WN'
    #Extract `poor accuracy`
    data['is_poor_accuracy'] = [1 if data['Accuracy'].iloc[i] > 5 else 0 for i in range(len(data))]
    #Extract negative value labeled from `Speed`
    data['is_negative_speed'] = [1 if data['Speed'].iloc[i] < 0 else 0 for i in range(len(data))]
    #Extract high-value labeled from `Speed`
    data['is_high_speed'] = [1 if data['Speed'].iloc[i] > 16 else 0 for i in range(len(data))]
    return data
    
def _kurtosis(x):
    return kurtosis(x)

def CPT5(x):
    den = len(x)*np.exp(np.std(x))
    return sum(np.exp(x))/den

def skewness(x):
    return skew(x)

def SSC(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1 
    xn_i1 = x[0:len(x)-2]  # xn-1
    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)
    return sum(ans[1:]) 

def wave_length(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1 
    return sum(abs(xn_i2-xn))
    
def norm_entropy(x):
    tresh = 3
    return sum(np.power(abs(x),tresh))

def SRAV(x):    
    SRA = sum(np.sqrt(abs(x)))
    return np.power(SRA/len(x),2)

def mean_abs(x):
    return sum(abs(x))/len(x)

def zero_crossing(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1
    return sum(np.heaviside(-xn*xn_i2,0))

def mean_change_of_abs_change(x):
    return np.mean(np.diff(np.abs(np.diff(x))))
    
def all_features(train):
    agg = pd.DataFrame()
    train = pd.get_dummies(train)
    for col in train.columns:
        if col in ['bookingID','bearing_bin','is_high_speed','is_poor_accuracy',
                  'bearing_bin_ES','bearing_bin_NE','bearing_bin_SW','bearing_bin_WN']:
            continue
        agg[str(col)+'_mean'] = train.groupby(['bookingID'])[col].mean()
        agg[str(col)+'_median'] = train.groupby(['bookingID'])[col].median()
        agg[str(col)+'_max'] = train.groupby(['bookingID'])[col].max()
        agg[str(col)+'_min'] = train.groupby(['bookingID'])[col].min()
        agg[str(col)+'_std'] = train.groupby(['bookingID'])[col].std()
        agg[str(col)+'_skew'] = train.groupby(['bookingID'])[col].skew()
        agg[str(col) + '_range'] = agg[str(col) + '_max'] - agg[str(col) + '_min']
        agg[str(col) + '_maxtoMin'] = agg[str(col) + '_max'] / agg[str(col) + '_min']
        agg[str(col) + '_mean_abs_chg'] = train.groupby(['bookingID'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        agg[str(col) + '_mean_change_of_abs_change'] = train.groupby('bookingID')[col].apply(mean_change_of_abs_change)
        agg[str(col) + '_abs_max'] = train.groupby(['bookingID'])[col].apply(lambda x: np.max(np.abs(x)))
        agg[str(col) + '_abs_min'] = train.groupby(['bookingID'])[col].apply(lambda x: np.min(np.abs(x)))
        agg[str(col) + '_abs_avg'] = (agg[col + '_abs_min'] + agg[col + '_abs_max'])/2
        agg[str(col)+'_mad'] = train.groupby(['bookingID'])[col].mad()
        agg[str(col)+'_q25'] = train.groupby(['bookingID'])[col].quantile(0.25)
        agg[str(col)+'_q75'] = train.groupby(['bookingID'])[col].quantile(0.75)
        agg[str(col)+'_q95'] = train.groupby(['bookingID'])[col].quantile(0.95)
        agg[str(col)+'_iqr'] = agg[str(col)+'_q75'] - agg[str(col)+'_q25']
        agg[str(col)+'_cpt5'] = train.groupby(['bookingID'])[col].apply(CPT5)
        agg[str(col)+'_ssc'] = train.groupby(['bookingID'])[col].apply(SSC)
        agg[str(col)+'_mean_abs'] = train.groupby(['bookingID'])[col].apply(mean_abs)
        agg[str(col)+'_skewness'] = train.groupby(['bookingID'])[col].apply(skewness)
        agg[str(col)+'_wave_length'] = train.groupby(['bookingID'])[col].apply(wave_length)
        agg[str(col)+'_norm_entropy'] = train.groupby(['bookingID'])[col].apply(norm_entropy)
        agg[str(col)+'_SRAV'] = train.groupby(['bookingID'])[col].apply(SRAV)
        agg[str(col)+'_kurtosis'] = train.groupby(['bookingID'])[col].apply(_kurtosis)
        agg[str(col)+'_zero_crossing'] = train.groupby(['bookingID'])[col].apply(zero_crossing)
    agg['len'] = train.groupby(['bookingID']).apply(len)
    agg['high_speed'] = train.groupby(['bookingID'])['is_high_speed'].sum() / agg['len']
    agg['negative_speed'] = train.groupby(['bookingID'])['is_negative_speed'].sum()
    agg['dist'] = train.groupby(['bookingID'])['Speed'].mean() * train.groupby(['bookingID'])['second'].max()
    agg['poor_accuracy'] = train.groupby(['bookingID'])['is_poor_accuracy'].sum() / agg['len']
    agg['bearing_bin_ES'] = train.groupby(['bookingID'])['bearing_bin_ES'].sum() / agg['len']
    agg['bearing_bin_NE'] = train.groupby(['bookingID'])['bearing_bin_NE'].sum() / agg['len']
    agg['bearing_bin_SW'] = train.groupby(['bookingID'])['bearing_bin_SW'].sum() / agg['len']
    agg['bearing_bin_WN'] = train.groupby(['bookingID'])['bearing_bin_WN'].sum() / agg['len']
    #Fix missing value and infinity value
    agg.fillna(0, inplace = True)
    agg.replace(-np.inf, 0, inplace = True)
    agg.replace(np.inf, 0, inplace = True)
    return agg
    
##Building train model
#Read train dataset and its label
train0 = pd.read_csv("../input/grabbb/part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
train1 = pd.read_csv("../input/outputtt/part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
train2 = pd.read_csv("../input/outputtt/part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
train3 = pd.read_csv("../input/outputtt/part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
train4 = pd.read_csv("../input/outputtt/part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
train5 = pd.read_csv("../input/outputtt/part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
train6 = pd.read_csv("../input/outputtt/part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
train7 = pd.read_csv("../input/outputtt/part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
train8 = pd.read_csv("../input/outputtt/part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
train9 = pd.read_csv("../input/outputtt/part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
train = pd.concat([train0,train1,train2,train3,train4,train5,train6,train7,train8,train9],axis=0)
del train0,train1,train2,train3,train4,train5,train6,train7,train8,train9,
label = pd.read_csv("../Documents/safety/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv")
label = pd.DataFrame(label.groupby('bookingID')['label'].sum())
label = label.reset_index()
train = preprocessing(train)
train_agg = all_features(train)
cols = ['Bearing_zero_crossing','gyro_tot_zero_crossing','Accuracy_zero_crossing',
        'tilt_angle_zero_crossing','acc_tot_zero_crossing','second_zero_crossing',
        'second_ssc','is_negative_speed_zero_crossing','roll_cpt5','tilt_angle_cpt5',
       'second_cpt5','Accuracy_cpt5','Bearing_cpt5','Speed_cpt5','pitch_cpt5','gyro_z_cpt5','gyro_y_cpt5']
train_agg.drop(cols,axis=1,inplace=True)
train_agg = pd.merge(train_agg,label,on='bookingID')
#Calculated permutation importance using XGBoost
X = train_agg.drop(['bookingID','label'],axis=1)
perm = PermutationImportance(xgb.XGBClassifier(), cv=skf)
perm.fit(X.values,y)
#put feature importances in dataframe
importances = pd.DataFrame()
importances['features'] = X.columns
importances['value'] = perm.feature_importances_
importances = importances.sort_values(by=['value'],ascending=False)
importances = importances.reset_index()
#Building weighted ensemble model
scaler = StandardScaler()
X = train_agg.drop(['bookingID','label'],axis=1)[importances[:150]['features']]
y = train_agg['label']
X_scaling = scaler.fit_transform(X)
model_xgb = xgb.XGBClassifier(n_estimators = 100)
model_lgb = lgb.LGBMClassifier()
model_lr = LogisticRegression()
model_rf = RandomForestClassifier()
model_xgb.fit(X_train,y_train)
model_lgb.fit(X_train,y_train)
model_lr.fit(X_train,y_train)
model_rf.fit(X_train,y_train)

#Read data test
test = pd.read_csv("../input_test.csv")
test = preprocessing(test)
test_agg = all_features(test)
test_agg = test_agg.reset_index()
cols = ['Bearing_zero_crossing','gyro_tot_zero_crossing','Accuracy_zero_crossing',
        'tilt_angle_zero_crossing','acc_tot_zero_crossing','second_zero_crossing',
        'second_ssc','is_negative_speed_zero_crossing','roll_cpt5','tilt_angle_cpt5',
       'second_cpt5','Accuracy_cpt5','Bearing_cpt5','Speed_cpt5','pitch_cpt5','gyro_z_cpt5','gyro_y_cpt5']
test_agg.drop(cols,axis=1,inplace=True)
X_test = test_agg.drop(['bookingID'],axis=1)[importances[:150]['features']]
predict_xgb = model_xgb.predict_proba(X_test)[:,1]*0.75
predict_lgb = model_lgb.predict_proba(X_test)[:,1]*0.10
predict_rf = model_rf.predict_proba(X_test)[:,1]*0.05
predict_lr = model_lr.predict_proba(X_test)[:,1]*0.10
tot = predict_xgb + predict_lgb + predict_rf + predict_lr 

#export prediction
result = pd.DataFrame()
result['bookingID'] = test_agg['bookingID']
result['probability'] = tot
result.to_csv("holdout_prediction.csv",index=False)


