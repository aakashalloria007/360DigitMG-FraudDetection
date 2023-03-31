import matplotlib as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from AutoClean import AutoClean
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error , accuracy_score ,confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from numpy import where
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
import imblearn
from sklearn.linear_model import LogisticRegression
import pickle
import joblib

data = pd.read_excel(r"C:\Users\aakas\Desktop\Data Science\DATA SCIENCE PROJECTS\Money_Laundering_Dataset.xlsx")
df = pd.DataFrame(data=data)

#EDA

df['isFraud'].value_counts()
df.info()
df.describe()
df.mean()
df.median()
df.mode()
df.var()
df.std()

range_amt = max(df.amount) - min(df.amount)
range_oldorig = max(df.oldbalanceOrg) - min(df.oldbalanceOrg) # range
range_neworig = max(df.newbalanceOrig) - min(df.newbalanceOrig)
range_olddest = max(df.oldbalanceDest) - min(df.oldbalanceDest)
range_newdest = max(df.newbalanceDest) - min(df.newbalanceDest) 

range_oldorig
range_neworig
range_olddest
range_newdest

df.skew()
df.kurt()

duplicate = df.duplicated()
duplicate
sum(duplicate)

df.drop(['Unnamed: 0','nameOrig','nameDest','isFlaggedFraud'],axis=1, inplace=True)
df.info()

df.shape
df.dtypes

#Fixing Missing Values

mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df["isFraud"] = pd.DataFrame(mode_imputer.fit_transform(df[["isFraud"]]))


#Check for correlation

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True)
df.head()

#Splitting The Data

X = df.iloc[:,:7]
Y = df.iloc[:,7]
X
Y


num_columns = X.select_dtypes(exclude=['object']).columns
num_columns

cat_columns = X.select_dtypes(include=['object']).columns
cat_columns

pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy= "median"))])
preprocessor = ColumnTransformer(transformers = [('num', pipeline, num_columns)])
imputation = preprocessor.fit(X)


imp_final = pd.DataFrame(imputation.transform(X), columns = num_columns)
imp_final

winsorizer = Winsorizer(capping_method='iqr', 
                        tail='both', 
                        fold=1.5,
                        variables=['step','oldbalanceOrg','newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

#Not Using Winsorizer on Amount as that can be used to get better Accuracy 

clean = winsorizer.fit(imp_final[['step','oldbalanceOrg','newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']])

df_clean = imp_final[['step','oldbalanceOrg','newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']] = clean.transform(imp_final[['step','oldbalanceOrg','newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']])
df_clean

minmaxscaler = Pipeline([['scale', MinMaxScaler()]])
scale_columntransformer = ColumnTransformer([('scale', minmaxscaler, num_columns)])
scaled_columns = scale_columntransformer.fit(X)
joblib.dump(scaled_columns,'minmaxscaler')

scaled_data = pd.DataFrame(scaled_columns.transform(imp_final))
scaled_data
scaled_data.describe()

encoding_pipeline = Pipeline([('onehot', OneHotEncoder())])
preprocess_pipeline = ColumnTransformer([('categorical', encoding_pipeline, cat_columns)])
clean = preprocess_pipeline.fit(X)

joblib.dump(clean,'onehotencoder')
encoded_data = pd.DataFrame(clean.transform(X).todense())
clean_data = pd.concat([scaled_data, encoded_data], axis=1 , ignore_index = True)
clean_data
x = clean_data


#Using Smote For Oversampling
over_sample = imblearn.over_sampling.SMOTE(random_state=0)
X,Y = over_sample.fit_resample(x,Y)
Counter(Y)

train_X_smot, test_X, train_Y_smot, test_y = train_test_split(X, Y, test_size = 0.3, random_state = 21)


train_X_smot
train_Y_smot


Counter(train_X_smot)


#SVC Model

model_linear = SVC(kernel = "linear",class_weight='balanced')
model1 = model_linear.fit(train_X_smot, train_Y_smot)
pred_test_svc = model_linear.predict(test_X)
print(classification_report(test_y,pred_test_svc))

#SVC HyperParametre Tuning

model = SVC()
parameters = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
rand_search =  RandomizedSearchCV(model, parameters, n_iter = 10, 
                                  n_jobs = -1, cv = 5, scoring = None, random_state = 1)
randomised = rand_search.fit(train_X_smot, train_Y_smot)
randomised.best_params_
best = randomised.best_estimator_
pred_test_svc_tuning = best.predict(test_X)

classification_report(test_y,pred_test_svc_tuning)

#RandomForestClassifier Model

classifier= RandomForestClassifier(n_estimators= 100, criterion="entropy")  
classifier.fit(train_X_smot, train_Y_smot)  
pred_test_RFC= classifier.predict(test_X)  
print(classification_report(test_y,pred_test_RFC))
pred_train_RFC= classifier.predict(train_X_smot)  
print(classification_report(train_Y_smot,pred_train_RFC))
#RandomForestClassifier HyperParametre Tuning


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
max_depth = [int(x) for x in np.linspace(10, 100, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
model = RandomForestClassifier()
rand_search =  RandomizedSearchCV(model, parameters, n_iter = 100, n_jobs = -1, cv = 3, scoring = None, random_state = 1)
randomised = rand_search.fit(train_X_smot, train_Y_smot)
randomised.best_params_
best = randomised.best_estimator_
pred_test_RFC_tuning = best.predict(test_X)
print(classification_report(test_y,pred_test_RFC_tuning))

#XGBBoost Model

model = xgb.XGBClassifier(
learning_rate =0.1,
n_estimators=500,
max_depth=5,
min_child_weight=1,
gamma=0,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary:logistic',
nthread=9,
scale_pos_weight=1,
seed=27,
tree_method='gpu_hist' )

train_model = model.fit(train_X_smot, train_Y_smot)  
pred_test_XGB = train_model.predict(test_X)
print(classification_report(test_y,pred_test_XGB))
print(confusion_matrix(test_y, pred_test_XGB))

#XGBBoost HyperParametre Tuning

folds = 2
param_comb = 20
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.02, 0.05]    
        }

from sklearn.model_selection import train_test_split ,GridSearchCV,StratifiedKFold,RandomizedSearchCV

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
xgbb = XGBClassifier(learning_rate=0.01, n_estimators=250, objective='binary:logistic',
                    silent=True, nthread=6, tree_method='gpu_hist', eval_metric='auc',warn = -1 )
random_search = RandomizedSearchCV(xgbb, param_distributions=params, n_iter=param_comb, n_jobs=4, cv=skf.split(train_X_smot,train_Y_smot),verbose=2, random_state=1001 )
train_model =random_search.fit(train_X_smot,train_Y_smot)
pred = train_model.predict(test_X)
print(train_model.best_params_)
print(train_model.best_estimator_)
cvres = train_model.cv_results_
best  = train_model.best_estimator_
pred = best.predict(test_X)
print(classification_report(test_y,pred))
pred_train = best.predict(train_X_smot)
print(classification_report(train_Y_smot,pred_train))

#KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(train_X_smot,train_Y_smot)
y_pred = knn.predict(test_X)
print(classification_report(test_y,y_pred))

y_pred_train = knn.predict(train_X_smot)
print(classification_report(train_Y_smot,y_pred_train))



#KNN HyperParametre

leaf_size = list(range(1,20))
n_neighbors = list(range(1,20))
p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
knn_2 = KNeighborsClassifier()
clf = GridSearchCV(knn_2, hyperparameters, cv=3,verbose=3,n_jobs=-1)
best_model = clf.fit(train_X_smot,train_Y_smot)
y_pred = best_model.predict(test_X)
print(classification_report(test_y,pred))



#LogicalRegressor

lr = LogisticRegression()
lr.fit(train_X_smot, train_Y_smot)
y_pred = lr.predict(test_X)
print(classification_report(test_y,y_pred))




from sklearn.model_selection import RepeatedStratifiedKFold
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers,penalty=penalty,C=c_values)
model = LogisticRegression()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=10, scoring='accuracy',error_score=0,verbose=2)
grid_result = grid_search.fit(train_X_smot,train_Y_smot)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
grid_result.best_params_
best = grid_result.best_estimator_
pred = best.predict(test_X)
print(classification_report(test_y,pred))


#KNN HyperParametre

leaf_size = list(range(1,20))
n_neighbors = list(range(1,20))
p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
knn_2 = KNeighborsClassifier()
clf = GridSearchCV(knn_2, hyperparameters, cv=3,verbose=3,n_jobs=-1)
best_model = clf.fit(train_X_smot,train_Y_smot)
y_pred = best_model.predict(test_X)
print(classification_report(test_y,pred))

pickle.dump(best_model,open('bestknn.pkl','wb'))


