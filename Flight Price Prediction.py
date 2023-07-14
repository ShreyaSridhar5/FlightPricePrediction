#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from xgboost import XGBRegressor


# In[2]:


train = pd.read_excel('Data_Train.xlsx')
test = pd.read_excel('Test_set.xlsx')


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


train.describe()


# In[6]:


train.isna().sum()


# In[7]:


train[train['Route'].isna()]


# In[8]:


train2 = train[train['Route'].notna()]


# Route, Total_Stops both have missing values. These lie in the same row, so we can safely drop it.

# In[9]:


def findUnique(column):
    print(column.name)
    print(column.unique())
    print('\n')
    
train2.apply(findUnique)


# ### Analysis of Values in each Column:
# 
# #### Airline: 
# Even within an airline, there is a division between economy and higher classes like Premium Economy/ Business. Would be interesting to look into if that difference is more of a dividing factor than the airline itself. 
# - make a new feature indicating class! 
# 
# #### Date of Journey: 
# Unfortunately, the data seems to be limited in scope to the months of March-June, and that too with quite a small selection of dates. Our analysis may fall short because of this. It may be interesting to extract what day of the week/ what month each flight took off on / landed on. 
# - try and find more data to increase our analysis ability! Make new feature to extract day of the week/month! 
# 
# #### Source/Destination: 
# We seem to be focusing on the largest cities in India. Could be interesting to: 
# - look into whether there is a correlation between the population of the cities and the prices associated with the flights there. 
# 
# #### Route: 
# The routes are written in short form. Should try and extract:
# - what the intermediate cities are. Price may depend on *which* cities a passenger is stopping over at. 
# 
# ####  Dep_Time: 
# Currently storing timing as strings. Is there a way to better represent time (ie in categories rather than as numbers? 
# - make a new feature indicating categories (overnight, early morning, morning, afternoon, evening, etc.)
# - Do certain times have more flights than others + does that impact price? 
# 
# #### Arrival Time: 
# Some values have a day written in the arrival, while some don't. This indicates an overnight flight. Does that impact price?
# - extract if a flight results in an overnight arrival. 
# 
# #### Duration: 
# Currently storing as strings. 
# - Convert to integers ( minutes). 
# 
# 
# 
# 

# In[10]:


def findNumAppear(column):
    print(column.name)
    print(column.value_counts())
    print('\n')
    
train2.apply(findNumAppear)


# ### Analysis of Number Times Each Value Appears
# 
# #### Airline:
# Jet Airways has the most appearances. There are relatively few appearances of routes that do not take economy class.
# - is the impact of class significant to keep it despite not having that much data?
# 
# #### Date of Journey:
# No dates appear much less than others. 
# - do certain months have less flights than others?
# 
# #### Source / Destination:
# Delhi has the most flights taking off, and Cochin has the most flights landing. 
# - can we find data regarding flights with Cochin as the source? It's likely that those numbers are quite high. 
# 
# #### Route: 
# - Extract what middle stops are and how frquently they occur
# 
# #### Dep/Arrival Time/Duration:
# - can we combine timings so we don't have so many times with only 1/2 flights 
# 
# #### Total Stops:
# - convert to numbers from strings
# 
# #### Additional Info
# - do we have enough info about all of these categories to have them add value to our model? 
# 

# ### Analyze Numerical Variables: 

# In[11]:


plt.hist(train['Price'])
plt.title('Price')
plt.show()


# Need to normalize the data -- skewed right now

# ### Analyze Categorical Variables

# In[12]:


categ_col = train[['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route',
       'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops',
       'Additional_Info']]
for i in categ_col.columns:
    sns.barplot(categ_col[i].value_counts().index, categ_col[i].value_counts()).set_title(i)
    plt.show()


# display graphs again after doing some editing of features 

# ### Data Cleaning/Feature Engineering:

# #### Airline

# In[13]:


train2['Airline'].value_counts()


# In[14]:


a = train2['Airline']== 'Multiple carriers Premium economy'
b = train2['Airline']== 'Vistara Premium economy'
c = train2['Airline']== 'Jet Airways Business'


# In[15]:


premium = a|b|c
premium[premium==False]=0
premium[premium==True]=1
train2['Premium'] = premium


# In[16]:


airline = train2['Airline']
train2['Airline'][airline == 'Multiple carriers Premium economy'] = 'Multiple carriers'
train2['Airline'][airline == 'Vistara Premium economy'] = 'Vistara'
train2['Airline'][airline == 'Jet Airways Business'] = 'Jet Airways'

train2['Airline'].value_counts()


# In[17]:


train2['Premium'].value_counts()


# In[18]:


print(pd.pivot_table(train2, index = 'Airline' , values = 'Price',aggfunc = np.mean))


# #### Date of Journey

# In[19]:


train2['Date_of_Journey'].value_counts()
newDate = pd.to_datetime(train2['Date_of_Journey'], format = '%d/%m/%Y')
                           


# In[20]:


sns.barplot(train2['Date_of_Journey'].value_counts().index, train2['Date_of_Journey'].value_counts()).set_title('Date_of_Journey')
plt.xticks(rotation=90)
plt.rcParams['figure.figsize'] = [5, 10]
plt.show()


# In[21]:


train2['DepMonth'] = newDate.dt.month
train2['DepWeekday'] = newDate.dt.dayofweek #0 = Monday, 6 = Sunday
train2['DepDay'] = newDate.dt.day


# In[22]:


train2['DepMonth'].value_counts()


# In[23]:


print(pd.pivot_table(train2, index = 'DepMonth' , values = 'Price',aggfunc = np.mean))


# April has significantly lower pricing.

# In[24]:


train2['DepWeekday'].value_counts()


# In[25]:


print(pd.pivot_table(train2, index = 'DepWeekday', values = 'Price',aggfunc = np.mean))
print()


# Mondays have the lowest prices.

# In[26]:


train2['DepDay'].value_counts()


# In[27]:


print(pd.pivot_table(train2, index = 'DepDay', values = 'Price',aggfunc = np.mean))
print()


# The 1st and 6th of the month seem to have the highest prices. The price tends to decrease as the month goes on.

# #### Source/Destination

# In[28]:


print(pd.pivot_table(train2, index = 'Source', values = 'Price',aggfunc = np.mean))
print()


# In[29]:


print(pd.pivot_table(train2, index = 'Destination', values = 'Price',aggfunc = np.mean))
print()


# Source: Chennai by far has the cheapest flights, and Delhi by far the most expensive.
# Destination: Delhi and Cochin are most expensive, and Kolkata is cheapest.

# In[30]:


print(pd.pivot_table(train2, index = 'Source', columns = 'Total_Stops',values = 'Price'))
print()


# Chennai only has non-stop flights, which explains why it's the cheapest. At each level of stops, Mumbai also has very expensive flights, but on the aggregate Delhi is more expensive. 

# In[31]:


print(pd.pivot_table(train2, index = 'Destination', columns = 'Total_Stops',values = 'Price'))
print()


# Again, New Delhi is the most expensive destination on the whole, but on individual levels there are other cities that are more expensive. Is this just because some cities just have more flights that are not non-stop than others?

# In[32]:


print(pd.pivot_table(train2, index = 'Source', columns = 'Total_Stops', values = 'Price',aggfunc = 'count'))


# In[33]:


print(pd.pivot_table(train2, index = 'Destination', columns = 'Total_Stops', values = 'Price',aggfunc = 'count'))


# Delhi has more flights that have multiple stops, explaining why it tends to be more expensive. 

# #### Route
# 
# -find what the intermediate cities are. Price may depend on which cities a passenger is stopping over at. 

# In[34]:


train2['Route'].value_counts()

def splitRoute(x):
    a =  x.split(' → ')
    
    if len(a)>2:
        return a[1:-1]

stops = train2['Route'].apply(splitRoute)

# cities = set()
# def getAllCities(x):
#     for i in x:
#         cities.add(i)

# (train2['Route'].apply(splitRoute)).apply(getAllCities)


# In[35]:


cities = set()
def getAllCities(x):
    if x!=None:
        for i in x:
            cities.add(i)

(train2['Route'].apply(splitRoute)).apply(getAllCities)


# In[36]:


cols = pd.DataFrame()

for i in cities:
    def hasCity(x):
        if x==None:
            return 0
        if i in x:
            return 1
        else:
            return 0 
    train2['Stopover' + i] = stops.apply(hasCity)


# In[37]:


train2.columns


# In[38]:


train2['Total_Stops'].value_counts()


# #### Stopovers

# In[39]:


for i in ['StopoverAMD', 'StopoverCOK', 'StopoverIMF', 'StopoverCCU',
       'StopoverHBX', 'StopoverSTV', 'StopoverBDQ', 'StopoverNDC',
       'StopoverJAI', 'StopoverVNS', 'StopoverKNU', 'StopoverIXZ',
       'StopoverIXR', 'StopoverIXA', 'StopoverIXC', 'StopoverPNQ',
       'StopoverATQ', 'StopoverBBI', 'StopoverTRV', 'StopoverGWL',
       'StopoverNAG', 'StopoverRPR', 'StopoverLKO', 'StopoverIXB',
       'StopoverBLR', 'StopoverGAU', 'StopoverISK', 'StopoverJLR',
       'StopoverIDR', 'StopoverMAA', 'StopoverGOI', 'StopoverUDR',
       'StopoverIXU', 'StopoverBOM', 'StopoverVTZ', 'StopoverPAT',
       'StopoverHYD', 'StopoverDED', 'StopoverJDH', 'StopoverVGA',
       'StopoverBHO', 'StopoverDEL']:
    
    print(i)
    print(pd.pivot_table(train2, index = i, values = 'Price',aggfunc = 'mean').iloc[1,])


# ####  Dep_Time: 
# Currently storing timing as strings. Is there a way to better represent time (ie in categories rather than as numbers? 
# - make a new feature indicating categories (overnight, early morning, morning, afternoon, evening, etc.)
# - Do certain times have more flights than others + does that impact price? 

# In[40]:


train2['Dep_Time']

def splitTime(x):
    return int(x.split(':')[0])

train2['HourDeparture'] = train2['Dep_Time'].apply(splitTime)


# In[41]:


print(pd.pivot_table(train2, index = 'HourDeparture', values = 'Price',aggfunc = 'mean'))


# In[42]:


train2['DeptNight'] = ((train2['HourDeparture']<=4) | (train2['HourDeparture']>20)).apply(int)
train2['DeptMorn'] = ((train2['HourDeparture']>4) & (train2['HourDeparture']<=12)).apply(int)
train2['DeptAfternoon'] = (train2['HourDeparture']>12) & (train2['HourDeparture']<=20).apply(int)


# In[43]:


for i in ['DeptMorn','DeptAfternoon','DeptNight']:
    print(i)
    print(pd.pivot_table(train2, index = i, values = 'Price',aggfunc = 'mean').iloc[1,])


# Doesn't seem to be a big difference between the various categories, night flights are the cheapest. 

# #### Arrival Time: 
# Some values have a day written in the arrival, while some don't. This indicates an overnight flight. Does that impact price?
# - extract if a flight results in an overnight arrival. 
# 

# In[44]:


train['Arrival_Time'].value_counts()


# In[45]:


train2['HourArrival'] = train2['Arrival_Time'].apply(splitTime)


# In[46]:


train2['HourArrival']


# In[47]:


print(pd.pivot_table(train2, index = 'HourArrival', values = 'Price',aggfunc = 'mean'))


# In[48]:


train2['ArrivNight'] = ((train2['HourArrival']<=4) | (train2['HourArrival']>20)).apply(int)
train2['ArrivMorn'] = ((train2['HourArrival']>4) & (train2['HourArrival']<=12)).apply(int)
train2['ArrivAfternoon'] = (train2['HourArrival']>12 & (train2['HourArrival']<=20)).apply(int)


# In[49]:


for i in ['ArrivNight','ArrivMorn','ArrivAfternoon']:
    print(pd.pivot_table(train2, index = i, values = 'Price',aggfunc = 'mean').iloc[1,])


# In[50]:


def containsLetters(x):
    if x.upper().isupper():
        return 1
    else:
        return 0
train2['landNextDay'] = train2['Arrival_Time'].apply(containsLetters)


# In[51]:


print(pd.pivot_table(train2, index = 'landNextDay', values = 'Price',aggfunc = 'mean'))


# Landing the next day has a significant impact on the price of the flights.

# #### Total Stops:
# - convert to numbers from strings

# In[52]:


train2['Total_Stops'].value_counts()


# In[53]:


train2['Total_Stops'][train2['Total_Stops']=='1 stop']=1
train2['Total_Stops'][train2['Total_Stops']=='2 stops']=2
train2['Total_Stops'][train2['Total_Stops']=='3 stops']=3
train2['Total_Stops'][train2['Total_Stops']=='4 stops']=4
train2['Total_Stops'][train2['Total_Stops']=='non-stop']=0


# In[54]:


print(pd.pivot_table(train2, index = 'Total_Stops', values = 'Price',aggfunc = 'mean').iloc[0:4,])


# #### Duration: 
# Currently storing as strings. 
# - Convert to integers ( minutes). 

# In[55]:


def calculateMinutes(x):
    hour = 0 
    minute = 0
    m = x.find('m')
    h = x.find('h')
    if h>0:
        hour = int(x[0:h])*60
    if m>0:
        try:
            minute = int(x[m-2:m])
        except: 
            minute = int(x[m-1:m])  
    return hour + minute

train2['DurationMinutes'] = train2['Duration'].apply(calculateMinutes)


# In[56]:


train2['DurationMinutes']


# In[57]:


plt.plot(pd.pivot_table(train2, index = 'DurationMinutes', values = 'Price',aggfunc = 'mean'))


# #### Additional Info
# - do we have enough info about all of these categories to have them add value to our model? 
# - combine 'Business class' with 'Premium'

# In[58]:


train2['Additional_Info'][train2['Additional_Info']== 'No info']='No Info'


# In[59]:


train2['Additional_Info'].value_counts()


# In[60]:


print(pd.pivot_table(train2, index = 'Additional_Info', values = 'Price',aggfunc = 'mean'))


# In[61]:


a = pd.get_dummies(train2['Additional_Info'], prefix = 'AddInfo').drop(labels = 'AddInfo_No Info',axis=1)
train2 = pd.concat([train2, a], axis=1)


# In[62]:


train2 = train2.drop(labels=['Additional_Info', 'Route', ], axis=1)


# In[63]:


train2['Premium'][train2['AddInfo_Business class']==1] = 1
train2 = train2.drop(labels = 'AddInfo_Business class', axis=1)


# - Would it be useful to make a column for long layovers, specifying 0/1/2 instead of having them be separated with the Add_Info columns? 
# - are the columns 'AddInfo_BusinessClass' and 'Premium'conveying the same information? 

# In[64]:


pd.set_option('display.max_columns',None)
train2.head()


#  TO DO: 
#  - drop date_of_journey, dep_time, arrival_time, duration
#  - one-hot encode source + destination + airline

# In[65]:


train2 = train2.drop(labels = ['Date_of_Journey','Dep_Time','Arrival_Time','Duration'], axis=1)
# drop these columns since already extracted info from them


# In[66]:


train2.head()


# In[67]:


for i in ['Source','Destination','Airline']:
    a = pd.get_dummies(train2[i], prefix = i)
    train2 = pd.concat([train2, a], axis=1)
    train2 = train2.drop(labels=i,axis=1)


# In[68]:


for i in train2.columns:
    plt.scatter(train2[i],train2['Price'])
    plt.title('' + i + ' v. Price')
    plt.xlabel(''+i)
    plt.ylabel('Price')
    plt.show()


# ### Normalizing Numerical Columns

# In[69]:


from sklearn import preprocessing
toNorm = ['DepWeekday','HourDeparture', 'HourArrival']
for i in toNorm:
    train2[i] = np.log(preprocessing.normalize([train2[i]])[0]+1)


# In[70]:


for i in toNorm:
    plt.hist(train2[i])
    plt.title('' + i + ' v. Price')
    plt.show()


# ### Model Selection w/ Cross Validation

# In[71]:


train2['Total_Stops'] = train2['Total_Stops'].astype('int')
train2['Premium'] = train2['Premium'].astype('int')
pd.set_option('display.max_rows', None)


# In[72]:


train2.columns


# In[73]:


Xtrain = train2[['Total_Stops', 'Premium', 'DepMonth', 'DepWeekday', 'DepDay',
       'StopoverBLR', 'StopoverTRV', 'StopoverHBX', 'StopoverJAI',
       'StopoverIXZ', 'StopoverBBI', 'StopoverIXB', 'StopoverKNU',
       'StopoverAMD', 'StopoverNDC', 'StopoverGAU', 'StopoverCOK',
       'StopoverBHO', 'StopoverBOM', 'StopoverIMF', 'StopoverVNS',
       'StopoverMAA', 'StopoverDEL', 'StopoverJLR', 'StopoverSTV',
       'StopoverHYD', 'StopoverJDH', 'StopoverVTZ', 'StopoverIXA',
       'StopoverGOI', 'StopoverLKO', 'StopoverRPR', 'StopoverPAT',
       'StopoverGWL', 'StopoverIXU', 'StopoverUDR', 'StopoverATQ',
       'StopoverDED', 'StopoverVGA', 'StopoverNAG', 'StopoverIDR',
       'StopoverIXC', 'StopoverISK', 'StopoverCCU', 'StopoverIXR',
       'StopoverBDQ', 'StopoverPNQ', 'HourDeparture', 'DeptNight', 'DeptMorn',
       'DeptAfternoon', 'HourArrival', 'ArrivNight', 'ArrivMorn',
       'ArrivAfternoon', 'landNextDay', 'DurationMinutes',
       'AddInfo_1 Long layover', 'AddInfo_1 Short layover',
       'AddInfo_2 Long layover', 'AddInfo_Change airports',
       'AddInfo_In-flight meal not included',
       'AddInfo_No check-in baggage included', 'AddInfo_Red-eye flight',
       'Source_Banglore', 'Source_Chennai', 'Source_Delhi', 'Source_Kolkata',
       'Source_Mumbai', 'Destination_Banglore', 'Destination_Cochin',
       'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata',
       'Destination_New Delhi', 'Airline_Air Asia', 'Airline_Air India',
       'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
       'Airline_Multiple carriers', 'Airline_SpiceJet', 'Airline_Trujet',
       'Airline_Vistara']]
Ytrain = train2['Price']


# In[74]:


#try RandomForest, Naive Bayes, SGDClassifier, SVM, KNN, Neural Network (tensorflow), XGBoost
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import make_scorer


# In[75]:


rf = RandomForestRegressor()
cv1=cross_val_score(rf,Xtrain,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cv1)
print(cv1.mean())


# In[76]:


sv = SVR()
cv3 = cross_val_score(sv,Xtrain,Ytrain, scoring = 'neg_mean_absolute_error',verbose=1)
print(cv3)
print(cv3.mean())


# In[77]:


knn = KNeighborsRegressor()
cv4 = cross_val_score(knn,Xtrain,Ytrain,scoring = 'neg_mean_absolute_error', verbose=1)
print(cv4)
print(cv4.mean())


# In[78]:


from xgboost import XGBRegressor
xgb = XGBRegressor(random_state = 1)
cv5 = cross_val_score(xgb,Xtrain,Ytrain,scoring = 'neg_mean_absolute_error',verbose=1)
print(cv5)
print(cv5.mean())


# In[79]:


mlp = MLPRegressor(max_iter = 700)
cv6 = cross_val_score(mlp,Xtrain,Ytrain,scoring = 'neg_mean_absolute_error',verbose=1)
print(cv6)
print(cv6.mean())


# ### Hyperparameter Tuning

# In[80]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# In[93]:


RFparams = {'n_estimators':[100,500,1000], 
          'max_depth':[None, 5,10,25,50,75,100],
          'max_features':['sqrt',0.3,0.5,0.75,1],
            'bootstrap':[True,False]}

rf2 = RandomForestRegressor()

grid_RF = RandomizedSearchCV(rf2, RFparams, scoring = 'neg_mean_absolute_error',verbose =1 )
grid_RF.fit(Xtrain,Ytrain)

print("RANDOM FOREST:")
print(grid_RF.best_score_)
print(grid_RF.best_params_)


# In[94]:


bestRF = RandomForestRegressor(n_estimators= 1000, max_features= 0.75, max_depth= 100, bootstrap= False)
bestRF.fit(Xtrain,Ytrain)


# In[95]:


forest_importances = pd.Series(bestRF.feature_importances_, index=Xtrain.columns).sort_values(ascending=False)
plt.figure(figsize=(20,6))
plt.bar(forest_importances.index[0:10], forest_importances[0:10])


# In[128]:


forest_importances[0:42]


# In[87]:


sv2 = SVR()
SVparams = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
            'degree':[3,5,7,9],
            'gamma':['scale','auto'],
            'C': [0.05,0.1,0.25,0.5,1]}

grid_SV = RandomizedSearchCV(sv2, SVparams, scoring = 'neg_mean_absolute_error',verbose =1)
grid_SV.fit(Xtrain,Ytrain)

print("Support Vector Regression:")
print(grid_SV.best_score_)
print(grid_SV.best_params_)


# In[88]:


knn2 = KNeighborsRegressor()
KNNparams = {'n_neighbors':[1,3,5,7,9,15,25],
            'weights':['uniform','distance'],
            'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
             'p':[1,2]}

grid_KNN = RandomizedSearchCV(knn2, KNNparams, scoring = 'neg_mean_absolute_error',verbose =1 )
grid_KNN.fit(Xtrain,Ytrain)

print("KNN:")
print(grid_KNN.best_score_)
print(grid_KNN.best_params_)


# In[89]:


xgb2 = XGBRegressor(random_state = 1)
XGBparams = {'booster':['gbtree','gblinear','dart'],
            'colsample_bytree':[0.2,0.5,0.8,1],
            'max_depth':[2,5,10,15,20,25,None],
             'learning_rate':[0.005, 0.01,0.05,0.1],
             'reg_alpha':[0,1,5,10,50,100],
             'reg_lambda':[0,1,5,10,50,100]
}

grid_XGB = RandomizedSearchCV(xgb2, XGBparams, scoring = 'neg_mean_absolute_error',verbose =1 )
grid_XGB.fit(Xtrain,Ytrain)

print("XGB:")
print(grid_XGB.best_score_)
print(grid_XGB.best_params_)


# In[92]:


mlp2 = MLPRegressor()
MLPparams = {'hidden_layer_sizes':[50,100,250],
             'activation':['identity', 'logistic', 'tanh', 'relu'],
             'solver':['lbfgs', 'sgd', 'adam'],
             'alpha':[0.0001,0.001,0.01,1,5,10],
             'batch_size':[10,50,200,500,'auto'],
             'learning_rate':['constant', 'invscaling', 'adaptive'],
             'max_iter':[200,500,700]
}

grid_MLP = RandomizedSearchCV(mlp2, MLPparams, scoring = 'neg_mean_absolute_error',verbose =1 )
grid_MLP.fit(Xtrain,Ytrain)

print("MLP:")
print(grid_MLP.best_score_)
print(grid_MLP.best_params_)


# ### Final Results

# #### Mean Squared Error of Results (using Cross Validation): 
# - Random Forest: -623.8715644457076
# - SVR: -2159.852682695897
# - KNN: -1658.5225190406004
# - XGB: -661.2359898635661
# - MLP: -1373.832693909264

# ### Feature Selection Using Random Forest

# In[110]:


featSelectX = Xtrain[forest_importances.index[0:42]]


# #### Random Forest: comparing the training of fewer features w/ baseline

# In[116]:


bestRF2 = RandomForestRegressor(n_estimators= 1000, max_features= 0.75, max_depth= 100, bootstrap= False)
cvfeatSelect=cross_val_score(rf,featSelectX,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cvfeatSelect)
print(cvfeatSelect.mean())


# In[123]:


bestRF2 = RandomForestRegressor(n_estimators= 1000, max_features= 0.75, max_depth= 100, bootstrap= False)
cvfeatSelect=cross_val_score(rf,Xtrain,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cvfeatSelect)
print(cvfeatSelect.mean())


# #### SVR: comparing the training of fewer features w/ baseline

# In[118]:


sv2 = SVR(kernel='linear', gamma='auto', degree= 3, C= 1)
cvfeatSelect=cross_val_score(sv2,featSelectX,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cvfeatSelect)
print(cvfeatSelect.mean())


# In[124]:


sv2 = SVR(kernel='linear', gamma='auto', degree= 3, C= 1)
cvfeatSelect=cross_val_score(sv2,Xtrain,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cvfeatSelect)
print(cvfeatSelect.mean())


# #### KNN: comparing the training of fewer features w/ baseline

# In[119]:


knn2 = KNeighborsRegressor(weights= 'uniform', p= 1, n_neighbors= 3, algorithm= 'brute')
cvfeatSelect=cross_val_score(knn2,featSelectX,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cvfeatSelect)
print(cvfeatSelect.mean())


# In[125]:


knn2 = KNeighborsRegressor(weights= 'uniform', p= 1, n_neighbors= 3, algorithm= 'brute')
cvfeatSelect=cross_val_score(knn2,Xtrain,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cvfeatSelect)
print(cvfeatSelect.mean())


# #### XGB: comparing the training of fewer features w/ baseline

# In[120]:


xgb2 = XGBRegressor(random_state = 1,reg_lambda= 50, reg_alpha= 100,
                                      max_depth= 25, learning_rate=0.1,
                                    colsample_bytree= 1, booster= 'gbtree')

cvfeatSelect=cross_val_score(xgb2,featSelectX,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cvfeatSelect)
print(cvfeatSelect.mean())


# In[126]:


xgb2 = XGBRegressor(random_state = 1,reg_lambda= 50, reg_alpha= 100,
                                      max_depth= 25, learning_rate=0.1,
                                    colsample_bytree= 1, booster= 'gbtree')

cvfeatSelect=cross_val_score(xgb2,Xtrain,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cvfeatSelect)
print(cvfeatSelect.mean())


# #### MLP: comparing the training of fewer features w/ baseline

# In[122]:


mlp2 = MLPRegressor(solver= 'adam', max_iter= 500, learning_rate='invscaling', 
                     hidden_layer_sizes= 50, batch_size= 10, alpha= 0.001, 
                    activation='relu')

cvfeatSelect=cross_val_score(mlp2,featSelectX,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cvfeatSelect)
print(cvfeatSelect.mean())


# In[127]:


mlp2 = MLPRegressor(solver= 'adam', max_iter= 500, learning_rate='invscaling', 
                     hidden_layer_sizes= 50, batch_size= 10, alpha= 0.001, 
                    activation='relu')

cvfeatSelect=cross_val_score(mlp2,Xtrain,Ytrain,scoring = 'neg_mean_absolute_error',verbose = 1,cv =5)
print(cvfeatSelect)
print(cvfeatSelect.mean())

