import pandas as pd
import numpy as np
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('superconduct/train.csv')
df2 = pd.read_csv('superconduct/unique_m.csv')


import matplotlib.pyplot as plt

# the histogram of the data

n, bins, patches = plt.hist(df['critical_temp'], density=True, facecolor='cyan', edgecolor='black')
plt.xlabel('Critical Temp(k)')
plt.ylabel('Density')
plt.title('Histogram of Tc')
plt.show()

#convert all the molecule count into 0/1
df_binary = df2.iloc[: , :-2] 
df_binary[df_binary != 0] = 1



df_mean = df_binary.mean(axis=0)
df_mean.sort_values(ascending=False, inplace=True)

#print(df_mean.index)

col_list = df_mean.index  #df_binary.columns

#print(col_list)


data = pd.DataFrame({'Molecule':df_mean.index, 'Mean':df_mean.values},index=col_list)


fig, ax = plt.subplots()
data.plot('Molecule', 'Mean', kind='scatter', ax=ax, c='red')
for k, v in data.iterrows():
   ax.annotate(k, v)
# set visibility of x-axis as False
xax = ax.axes.get_xaxis()
xax = xax.set_visible(False)

plt.show()



# find average ctical temperature for molecules
data_full = df_binary.join(df2['critical_temp'])

dict_temp = {}
for (columnName, columnData) in data_full.iteritems():
    rslt_df = data_full[(data_full[columnName] == 1)]    
    if not rslt_df.empty and columnName != 'critical_temp':
        dict_temp[columnName] = rslt_df["critical_temp"].mean()



series = pd.Series(dict_temp)
series.sort_values(ascending=False, inplace=True)
data2 = pd.DataFrame({'Molecule':series.index, 'Tc_Mean':series.values},index=series.index.tolist())

#plot mean critical temperature

fig, ax2 = plt.subplots()
data2.plot('Molecule', 'Tc_Mean', kind='scatter', ax=ax2, c='blue')
for k, v in data2.iterrows():
   ax2.annotate(k, v)

plt.ylabel('Mean critical temperature')

xax = ax2.axes.get_xaxis()
xax = xax.set_visible(False)
plt.show()


#### linear model to predict unique formula



df2.fillna(0)

x_data = df2.iloc[: , :-2]
y_data = df2['critical_temp']

#Splitting the dataset into training(70%) and test(30%)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,
                                                    random_state=66)


#Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

Y_prediction = lin_reg.predict(X_test)
print(Y_prediction) #-->predict the data

#checking accuracy
# importing r2_score module

# predicting the accuracy score
score=r2_score(y_test,Y_prediction)
print('r2 socre is ',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,Y_prediction))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,Y_prediction)))



#feature importance/dimentionality reduction
#from train data set
#first split the dataset into train/test by 70/30 %

from sklearn.decomposition import PCA


df.fillna(0)

X = df.iloc[: , :-1]
y = df['critical_temp']

#Splitting the dataset into training(70%) and test(30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                     random_state=66)



#scale predictor variables
pca = PCA()
X_reduced = pca.fit_transform(scale(X))


plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
plt.title('Cumulative explained variance by number of principal components', size=10)
plt.show()


#define cross validation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

regr = LinearRegression()
mse = []



# Calculate MSE with only the intercept
score = -1*model_selection.cross_val_score(regr,
           np.ones((len(X_reduced),1)), y, cv=cv,
           scoring='neg_mean_squared_error').mean()    
mse.append(score)

principal_component = 30

# Calculate MSE using cross-validation, adding one component at a time
for i in np.arange(1, principal_component):
    score = -1*model_selection.cross_val_score(regr,
               X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)
    
# Plot cross-validation results    
plt.plot(mse)
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title('Tc')
plt.show()

#form the above plt got 25 is the best number of features


#updated final principal components
principal_component = 25

print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))


#now evaluate 4 models on the same principal component and see the performance

#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

#scale the training and testing data

X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.transform(scale(X_test))[:,:principal_component]

#train PCR model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:principal_component], y_train)

print('LinearRegression score 25 - ', regr.score(X_reduced_test, y_test))


#calculate RMSE lineare regression with PCR
pred = regr.predict(X_reduced_test)
print("RMSE: linear regression with PCR")
print(np.sqrt(mean_squared_error(y_test, pred)))


#Accuracy check for different regression models with different number of features
accuracy = []
ridge_accuracy = []
lasso_accuracy = []
elastic_accuracy = []
# try principal_component from 5 to 50
n_PCR = [5,10,15,20,25,30,35,40,45,50,55,60]
for features in n_PCR:
	#principal_component = 11
	X_reduced_train = pca.fit_transform(scale(X_train))
	X_reduced_test = pca.transform(scale(X_test))[:,:features]

	#train PCR model on training data 
	regr = LinearRegression()
	regr.fit(X_reduced_train[:,:features], y_train)
	accuracy.append(regr.score(X_reduced_test, y_test)) 
	clf = Ridge(alpha=1.0)
	clf.fit(X_reduced_train[:,:features], y_train)
	ridge_accuracy.append(clf.score(X_reduced_test, y_test))
	cld = linear_model.Lasso(alpha=0.1)
	cld.fit(X_reduced_train[:,:features], y_train)
	lasso_accuracy.append(cld.score(X_reduced_test,y_test))
	elastic = ElasticNet(random_state=0)
	elastic.fit(X_reduced_train[:,:features], y_train)
	elastic_accuracy.append(elastic.score(X_reduced_test, y_test))

#plt.plot(n_PCR, accuracy,label="Multiple LR", color="cyan", );
plt.plot(n_PCR, accuracy,label="Multiple LR", color="red",linestyle='solid');
plt.plot(n_PCR,elastic_accuracy,label="ElasticNet",color="cyan",linestyle='dashed' ) 
plt.plot(n_PCR,ridge_accuracy,label="Ridge",color="gray",linestyle='dashed');
plt.plot(n_PCR,lasso_accuracy,label="Lasso",color="blue",linestyle='dashed');
plt.ylabel("Accuracy");
plt.xlabel("Number of Features");
plt.legend();
plt.grid();
#saving the plot figure instead of showing. check the directory to find the plot image
plt.savefig('Linear_Regression_Comparison.png',dpi = 300)

# First generate all the principal components
pca = PCA()
#principal_component = 11
X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.transform(scale(X_test))[:,:principal_component]


#Run Linear regression model with top 25 features

# model = sm.OLS(y_train, X_reduced_train[:,:principal_component]).fit()
# print(model.summary())



#model evaluation
# Run standardization on X variables
X_train_scaled, X_test_scaled = scale(X_train), scale(X_test)

# Define cross-validation folds
cv = KFold(n_splits=10, shuffle=True, random_state=42)

lin_reg = LinearRegression().fit(X_train_scaled, y_train)
# Get R2 score
lin_reg.score(X_train_scaled, y_train)
lr_scores = -1 * cross_val_score(lin_reg, 
                                 X_train_scaled, 
                                 y_train, 
                                 cv=cv, 
                                 scoring='neg_root_mean_squared_error')
print('lr_scores - ',lr_scores)

lr_score_train = np.mean(lr_scores)
print('lr_score_train - ', lr_score_train)

y_predicted = lin_reg.predict(X_test_scaled)
lr_score_test = mean_squared_error(y_test, y_predicted, squared=False) # RMSE instead of MSE
print('lr_score_test - ', lr_score_test)


#(2) Lasso Regression (L1 regularization)

lasso_reg = LassoCV().fit(X_train_scaled, y_train)
lasso_scores = -1 * cross_val_score(lasso_reg, 
                                    X_train_scaled, 
                                    y_train, 
                                    cv=cv, 
                                    scoring='neg_root_mean_squared_error')
print('lasso_scores - ', lasso_scores)
lasso_score_train = np.mean(lasso_scores)
lasso_score_train


y_predicted = lasso_reg.predict(X_test_scaled)
lasso_score_test = mean_squared_error(y_test, y_predicted, squared=False)
print('lasso_score_test - ', lasso_score_test)


#(3) Ridge Regression (L2 regularization)

ridge_reg = RidgeCV().fit(X_train_scaled, y_train)

# Get R2 score
ridge_reg.score(X_train_scaled, y_train)

ridge_scores = -1 * cross_val_score(ridge_reg, 
                                    X_train_scaled, 
                                    y_train, 
                                    cv=cv, 
                                    scoring='neg_root_mean_squared_error')
print('ridge_scores - ', ridge_scores)

ridge_score_train = np.mean(ridge_scores)
print('ridge_score_train - ', ridge_score_train)

y_predicted = ridge_reg.predict(X_test_scaled)
ridge_score_test = mean_squared_error(y_test, y_predicted, squared=False)
print('ridge_score_test - ', ridge_score_test)



# Principal Components Regression
# Evaluate for different number of principal components


# Train model on training set

# First generate all the principal components
pca = PCA()
X_train_pc = pca.fit_transform(X_train_scaled)



lin_reg_pc = LinearRegression().fit(X_train_pc[:,:principal_component], y_train)

# Get R2 score
lin_reg_pc.score(X_train_pc[:,:principal_component], y_train)

pcr_score_train = -1 * cross_val_score(lin_reg_pc, 
                                       X_train_pc[:,:principal_component], 
                                       y_train, 
                                       cv=cv, 
                                       scoring='neg_root_mean_squared_error').mean()
print('pcr_score_train - ',pcr_score_train)



# Get principal components of test set
X_test_pc = pca.transform(X_test_scaled)[:,:principal_component]


# Predict on test data
preds = lin_reg_pc.predict(X_test_pc)
pcr_score_test = mean_squared_error(y_test, preds, squared=False)
print('pcr_score_test - ', pcr_score_test)

#Evaluation of model performance on train dataset and test dataset


test_metrics = np.array([round(lr_score_test,3), 
                          round(lasso_score_test,3), 
                          round(ridge_score_test,3), 
                          round(pcr_score_test,3)]) 
test_metrics = pd.DataFrame(test_metrics, columns=['RMSE (Test Set)'])
test_metrics.index = ['Linear Regression', 
                       'Lasso Regression', 
                       'Ridge Regression', 
                       f'PCR ({principal_component} components)']
print('test_metrics - ')
print(test_metrics)


train_metrics = np.array([round(lr_score_train,3), 
                          round(lasso_score_train,3), 
                          round(ridge_score_train,3), 
                          round(pcr_score_train,3)]) 
train_metrics = pd.DataFrame(train_metrics, columns=['RMSE (Train Set)'])
train_metrics.index = ['Linear Regression', 
                       'Lasso Regression', 
                       'Ridge Regression', 
                       f'PCR ({principal_component} components)']
print('train_metrics - ')
print(train_metrics)

#barplot for model score comparison

Model = ['MLR','Ridge','Lasso','PCR(25)']
RMSE = [lr_score_test,ridge_score_test,lasso_score_test,pcr_score_test]
plt.bar(Model,RMSE, color ='green')
plt.ylabel("RMSE of Critical Temperature (K)");

#saving the plot figure instead of showing. check the directory to find the plot image
plt.savefig('barplot.png', dpi = 300)

