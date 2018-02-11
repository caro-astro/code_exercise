#non distributed thoughts
#import statements
import numpy as np
import pandas as pd
from scipy import stats, integrate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 

#task -read the data frame (nonspark) automatically, 

file1=~/folder/dataframe.csv
thedf=pd.read_csv=(file1)

#or if sql: pd.read_sql_table()

#task - and determine the number of rows, columns, and numeric/categorical features.
#pandas automatically does this but to check: 
dfshape=pd.shape(thedf) #apparently spark doesn't have the shape command yet. 
ncols=dfshape[1]
nrows=dfshape[0]
df_column_names=pd.columns(thedf)

#note: your description of the table said:"Each person has more than one row 
#with relevant information for each month spanning over 4 years. (So for 4 years 
#each person may have up to 48 rows)" -- The groupby funcion df.groupby('SSID').agg({}) using the social security number (or other identifying key)
#should be used here, but I need more time to think about how the final table should look. Going forward, I will
#program as if this has been sorted out properly.
dfgroup=thedf after grouping

#task -automatically clean the data, i.e. determine outliers, anomalies, NULLs, 
#and map categorical features to dummy numeric variables and normalize numeric features for better modeling.

#I'd like to check the distributions of the data graphically before clipping outliers. To do that I would check 
#important variables using plots.
#plotting the distribution of chinesefood on y vs. month on the x axis
y=dfgroup["chinesefoodcost"]
x=dfgroup["month"]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="k")
plt.show()


#check for outliers stastically:
stats_sum=df.describe(dfgroup)
#choose a threshold in standard deviation: if it is a normal distribution then we can cut at 3-sigma to exclude outliers
lowerbound=mean-(3*sigma)
upperbound=mean+(3*sigma)
new_column=where(chinesefoodcost lt upperbound and chinesefoodcost gt lowerbound) # not python, will fix later
# do this for all relavent columns (or use a faster way that I don't know :)
#make a new data frame with good values
dfclean=dfgroup without outliers

#is scaling necessary for the dataset? probably, but graphing the KDE will show if the mean and std are the same for several variables or not
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
plt.show()

#to scale
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(dfclean)
scaled_df = pd.DataFrame(scaled_df, columns=['x1', 'x2', 'x3'])

-Using a standard ML algorithms (XGboost, Naive Bayes, LMST, KNN, Decision Tree…etc), predict “how much the person will spend on Chinese food” in 3 months.  Preferably apply at least three different predictive approaches and compare at the end.
#I am going to copy paste a random forest program from some previous code here because I am running out of time.
#splitting data into training and test sets
y = scaled_df.chinesefoodcost
X = scaled_df.drop('chinesefoodcost', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)
# data preprocessing steps: #I already scaled in a previous step, but scaling could also be done here. Standard scaler is good for normal distributions,
#MinMax scaler is better for non-Normal, and if I hadn't removed the outliers in another way, I could use robust scaler, which handles
#outliers better.

pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))
 
# hyperparameters for tuning:
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
#Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)


-Construct at least one Deep Learning approach to make the same prediction.
#????????? this is not something I have experience with unfortunately.

-Compare performance of all predictions 

#Run model pipeline on test data, and 
pred = clf.predict(X_test)
score_r2=r2_score(y_test, pred)
err_rms=mean_squared_error(y_test, pred)




