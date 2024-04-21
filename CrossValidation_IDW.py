#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import collections
import os
import arcpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (16, 12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)


# In[2]:


# set geoprocessing environments
arcpy.env.workspace = r"C:\Users\Owner\Desktop\g670wk12"
arcpy.env.overwriteOutput = True    
from arcpy import env
from arcpy.sa import *


# In[3]:


# Load - add Spatial Analyst
arcpy.CheckOutExtension("Spatial")
outputdir= r"C:\Users\Owner\Desktop\g670wk12"


# In[4]:


# Define training and validation shape files
trainClass = r"C:\Users\Owner\Desktop\g670wk12\Stony_topo_train\Stony_topo_train.shp"
# validClass has to be replaced at each run because it will be updated at each run
# If it is not replaced, it will contain 2 times more fields, which will make the parameter list 
# and the RMSE list have different numbers, breaking the running sequence
validClass = r"C:\Users\Owner\Desktop\g670wk12\s_valid\s_valid.shp"

# Define the interpolation variable
varList = ['X', 'Y', 'Z']
varName = varList[2]
# print(varName)

# Define the parameters for the power and neighborhood parameters
powerVal = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6] 
pointsNum =  [4, 6, 8, 10, 12, 14, 16, 22, 24]

# Define a list to save the interpolation outcomes
tiffList = []

# Define a list to save the power and neighbors indices
ppindex = []


# In[5]:


# Perform the IDW interpolation
kk=0
for ii in powerVal:
    kk=kk+1
    for jj in pointsNum:
        #outSpline = arcpy.sa.Spline(featureClass, k, 250, "REGULARIZED", ii, jj)
        #outSpline.save("C:/Kite_Folder/Spline_output_2013_0712_0728/"+str(k)+"_"+str(ii)+"_"+str(jj)+".tif")
        #print (varName+str(ii)+"_"+str(jj)+" is done")
        #Idw(in_point_features, z_field, {cell_size}, {power}, {search_radius}, {in_barrier_polyline_features})
        
        outIDW = arcpy.sa.Idw(trainClass, varName, 5.0, ii, RadiusVariable(jj, 40)) 
        # outIDW.save("C:/Users/yxie/TEACH/g585/2023_Win/Week11/"+varName+str(kk)+"_"+str(jj)+".tif")
        output_path = os.path.join(outputdir, f"{varName}{ii}_{jj}.tif")
        outIDW.save(output_path)
        # print(varName+str(ii)+"_"+str(jj)+" is done")
        tiffList.append(outIDW)
        ppindex.append(varName+str(ii)+"_"+str(jj))
      
arcpy.sa.ExtractMultiValuesToPoints(validClass, tiffList)
arcpy.CheckInExtension("Spatial")
print(ppindex)


# In[6]:


# Get the values for each field from the validation shapefile into tuples
field_names = [f.name for f in arcpy.ListFields(validClass)]
records_fc = set()
with arcpy.da.SearchCursor(validClass, field_names) as cursor:
    for row in cursor:
        records_fc.add(tuple(row))
#print(records_fc)


# In[7]:


# Concert tuples into lists
dataList = []
dataList = [*records_fc,]

# Convert the list into dataframe
rmseData = pd.DataFrame(list(dataList))
#print(rmseData)


# In[8]:


# Create a dataset to store RMSE
rmseSet = []
i = 0

# the observed value
Yobs = rmseData[2]
# print(Yobs)

# the intrpolated values
# Xint = rmseData[3:]
# print(Xint)

# Get rid of the 1st two columns
df = pd.DataFrame(rmseData)
dfT1 = df.T
dfT2 = dfT1[3:]
Xint = dfT2.T
# print(Xint)

# Compute the RMSE
for x in Xint:
    i=i+1
    # MSE = mean_squared_error(Yobs, x)
    # rmseSet.append(MSE)
    MSE = np.square(np.subtract(Yobs,x)).mean()
    RMSE = math.sqrt(MSE)
    # print("Root Mean Square Error:\n")
    # print(RMSE)
    rmseSet.append(RMSE)
    
# print(rmseSet)
# rmseSet.sort()
# print(rmseSet)

# Plot RMSE
plt.figure(figsize=(16,2))
plt.plot(rmseSet);
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'));
plt.title('Root Mean Squre Errors', fontsize=18);


# In[9]:


# Write the results as a file
fileName = 'RMSEwithIndex'
Abspath = r"C:\Users\Owner\Desktop\g670wk12"
#print(ppindex)
#print("\n")
#print(rmseSet)
writer=pd.ExcelWriter(Abspath+fileName+'.xlsx')
d = {'PowerNeighbor': ppindex, 'RMSE': rmseSet}
df=DataFrame(data=d)
print(df)
df.to_excel(writer,'Sheet1')
writer.close()
print('success')


# In[ ]:





# In[ ]:





# In[10]:


# Import packages
# Using the root environments
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[11]:


# Importing training data set
# Showing the data
X_train=pd.read_csv(r"C:\Users\Owner\Desktop\geog670Week13\X_train.csv")
Y_train=pd.read_csv(r"C:\Users\Owner\Desktop\geog670Week13\Y_train.csv")
# Importing testing data set
X_test=pd.read_csv(r"C:\Users\Owner\Desktop\geog670Week13\X_test.csv")
Y_test=pd.read_csv(r"C:\Users\Owner\Desktop\geog670Week13\Y_test.csv")


# In[19]:


# Lets take a closer look at our data set.
print (X_train.head())


# In[18]:


# Numerical Feature Exploration
X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])


# In[20]:


# Importing MinMaxScaler and initializing it
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
# Scaling down both train and test data set
# Why we are using numerical variables
X_train_minmax=min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax=min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])


# In[21]:


# Fitting k-NN on our scaled data set
# Classifier implementing the k-nearest neighbors vote
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_minmax,Y_train)
# Checking the model's accuracy
accuracy_score(Y_test,knn.predict(X_test_minmax))


# In[22]:


# Standardizing the train and test data
# What is the difference between Standardizing scale and MinMax scale
from sklearn.preprocessing import scale
X_train_scale=scale(X_train[['ApplicantIncome', 'CoapplicantIncome',
               'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_scale=scale(X_test[['ApplicantIncome', 'CoapplicantIncome',
               'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

# Fitting logistic regression on our standardized data set
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(penalty='l2',C=.01)
log.fit(X_train_scale,Y_train)
# Checking the model's accuracy
accuracy_score(Y_test,log.predict(X_test_scale))


# In[26]:


# Importing LabelEncoder and initializing it
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Iterating over all the common columns in train and test
for col in X_test.columns.values:
    # Encoding only categorical variables
    if X_test[col].dtypes == 'object':
        # Using whole data to form an exhaustive list of levels
        data = pd.concat([X_train[col], X_test[col]], axis=0)  # Concatenate Series into a DataFrame
        le.fit(data.values)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])


# In[27]:


print(X_train.head())
print(X_train.tail())


# In[28]:


# Logistic regression
# Standardizing the features

X_train_scale=scale(X_train)
X_test_scale=scale(X_test)

# Fitting the logistic regression model

log=LogisticRegression(penalty='l2',C=.01)
log.fit(X_train_scale,Y_train)

# Checking the models accuracy
accuracy_score(Y_test,log.predict(X_test_scale))


# In[32]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)

# Creating copies of X_train and X_test
X_train_1 = X_train.copy()
X_test_1 = X_test.copy()

columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
           'Credit_History', 'Property_Area']

# Iterating through categorical columns
for col in columns:
    # Creating an exhaustive list of all possible categorical values
    data = pd.concat([X_train[[col]], X_test[[col]]], axis=0)
    enc.fit(data)

    # Fitting One Hot Encoding on train data
    temp_train = enc.transform(X_train[[col]])
    temp_train = pd.DataFrame(temp_train, columns=[(col + "_" + str(i)) for i in range(temp_train.shape[1])])
    temp_train = temp_train.set_index(X_train.index.values)
    X_train_1 = pd.concat([X_train_1, temp_train], axis=1)

    # Fitting One Hot Encoding on test data
    temp_test = enc.transform(X_test[[col]])
    temp_test = pd.DataFrame(temp_test, columns=[(col + "_" + str(i)) for i in range(temp_test.shape[1])])
    temp_test = temp_test.set_index(X_test.index.values)
    X_test_1 = pd.concat([X_test_1, temp_test], axis=1)


# In[33]:


print(X_train_1.head())
print(X_train_1.tail())


# In[34]:


# Now, lets apply logistic regression model on one-hot encoded data.
# Standardizing the data set
X_train_scale=scale(X_train_1)
X_test_scale=scale(X_test_1)

# Fitting a logistic regression model
log=LogisticRegression(penalty='l2',C=1)
log.fit(X_train_scale,Y_train)

# Checking the model's accuracy
accuracy_score(Y_test,log.predict(X_test_scale))


# In[35]:


# Building Decision Tree
# Standardizing the data set
X_train_scale=scale(X_train_1)
X_test_scale=scale(X_test_1)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dt.fit(X_train_scale, Y_train)
accuracy_score(Y_test,dt.predict(X_test_scale))
text_representation = tree.export_text(dt)
print(text_representation)
with open("C:/Users/Owner/Desktop/geog670Week13/decistion_tree.log", "w") as fout:
    fout.write(text_representation)


# In[36]:


import graphviz
# DOT data
dot_data = tree.export_graphviz(dt, out_file=None, 
                                feature_names=X_train_1.columns.values,  
                                class_names=X_train_1.columns.values,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph


# In[37]:


plt.figure(figsize=(25,20))
# _ = tree.plot_tree(dt, 
#                    feature_names=X_train_scale.feature_names,  
#                   class_names=Y_train_scale.target_names,
#                   filled=True)
_ = tree.plot_tree(dt, 
                   feature_names=X_train_1.columns.values,  
                   class_names=X_train_1.columns.values,
                   filled=True)


# In[ ]:




