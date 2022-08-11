#Necessary Libraries
import pandas as pd
from sklearn import linear_model
from sklearn.cluster import KMeans
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Load in dataset into dataframe, preview it to confirm and understand what variables are
df = pd.read_csv("C:/Users/tirum/Downloads/data_kaggle.csv")
df.head(10)

#Check the number of rows and number of null rows
df.info()



#CLEANING
#Cleaning Price
#1. Remove nulls, because otherwise it's either a free property or doesn't exist and storing as a new dataframe to be safe
props = df.dropna(subset = ['Price'])
#2. Remove currency format
props.Price = props.Price.apply(lambda x: x.replace("RM ", "").replace(",", ""))
props.head(10)
#3. Make prices numeric
props.Price = pd.to_numeric(props.Price)

#Cleaning Rooms
#1. Split around + sign as delimiter into two columns (one being current Rooms column, the other being a new column that we will remove later), preview to confirm
props[["Rooms", "Additional to Remove"]] = props["Rooms"].str.split("+", expand = True)
props.head(10)
props.head(10)
#2.Remove aforementioned new column, preview to confirm
props.drop('Additional to Remove', axis=1, inplace=True)
props.head(10)
#3. Make rooms numeric [ERROR MESSAGE VERSION]
props.Rooms = pd.to_numeric(props.Rooms)
#4. Coerce errors to remove string entries, rename column to just Bedrooms, preview to check changes
props.Rooms = pd.to_numeric(props.Rooms, errors = 'coerce')
props = props.dropna(subset = ['Rooms'])
props.rename(columns = {'Rooms':'Bedrooms'}, inplace = True)
props.head(10)
#Ensure too many observations were not lost
props.info()

#Cleaning Size
#1. Split around : as delimiter into two columns (one being current Size column, the other being a new categorical column that we will also keep), preview to confirm
props[['Size_Type', 'Size']] = props.Size.str.split(" : ", expand = True)
props.head(10)
#2. Remove nulls, because otherwise a property physically does not exist
props = props.dropna(subset = ['Size'])
props.sample(10)
#3. Define a function to remove the various irregular characters and multiply the Size entries written as products of dimensions [ERROR MESSAGE VERSION WRITTEN AS COMMENT] / Define a function to remove the various irregular characters and convert what can be converted to numeric format
def clean(data, to_scrap):
    data['Size'] = data['Size'].apply(str)
    data = data[data['Size'].str.contains(r'\d')]
    for character in to_scrap:
        data = data[~data['Size'].str.contains(character, regex=False)]
    data['Size'] = data['Size'].apply(lambda x: x.replace('sq. ft.', '').replace('sq.ft.', '').replace('sq ft', '').replace(' sf', '').replace('sf', '').replace('ft', '').replace('sqft', '').replace(' sq. m.', '').replace(',', '').replace("'","").replace('`','').replace('approx',''))
    data['Size'] = data['Size'].apply(lambda x: x.replace('x', '*').replace(' x', '*').replace('x ', '*').replace('xx', '*').replace('@', '*').replace('X', '*').replace('/', '*'))
    data['Size'] = data['Size'].apply(lambda x: x.strip())
    data['Size'] = pd.to_numeric(data['Size'], errors='ignore')
    #data['Size'] = data['Size'].apply(lambda x: eval(x))
    return data
scrap = ['acre', 'Acre', 'arce', 'hectare', '~', '&', '#', 'wt',', ']
props = clean(props, scrap)
props.info()
props.sample(10)
#4. View how many entries will be lost given unsuccessful evaluation of products of dimensions as Size values
asterisks = props[props.Size.str.contains("*", regex=False) | props.Size.str.contains('-', regex=False) | props.Size.str.contains('+', regex=False)]
asterisks.info()
#5. Define a function to convert Size entries to numeric format, coercing errors and dropping those non-numeric rows. View the dataframe information and preview a random 10 rows to see these changes and their effects.
def final_clean(data):
    data['Size'] = pd.to_numeric(data['Size'], errors='coerce')
    return data
props = final_clean(props)
props = props.dropna(subset = ['Size'])
props.info()
props.sample(10)



#REGRESSION
#Ensuring property area is only measured as "Built-up" or "Land area'
props.Size_Type.unique()

#Selecting only built-up properties in a new dataframe for this work specifically
built = props[props.Size_Type == "Built-up"]
built.sample(10)

#Checking correlations before picking regressions out of instinct
sns.heatmap(built.corr(), annot = True, cmap='Reds');

#Saving a new dataframe with only the numeric columns for convenience, understanding the spread, average, and size of the data
numeric = built[['Price', 'Bedrooms', 'Bathrooms', 'Car Parks', 'Size']]
numeric.describe()

#Ignoring properties below 500 square feet that are unrealistic, understanding the spread, average, and size of the data after this change
built = built[built.Size >= 500]
numeric = numeric[numeric.Size >= 500]
numeric.describe()

#Visualizing boxplots to see the spread [POOR AXES]
sns.boxplot(data = numeric); 

#Visualizing boxplots with more detail
sns.boxplot(data = numeric);
plt.ylim(0, 100000);

#Creating new columns for the transformations of Size and Price
LN_Price = np.log(numeric.Price)
LN_Size = np.log(numeric.Size)
numeric['LN(Price)'] = LN_Price
numeric['LN(Size)'] = LN_Size
numeric.describe()

#Heatmap for new possible correlations
sns.heatmap(numeric.corr(), annot = True, cmap = 'Reds');

#Boxplots using transformed data [OVERLAPPING X AXIS LABELS]
sns.boxplot(data = numeric);
plt.ylim(0,25);

#Failed attempt to enlarge plot to resolve overlapping x axis labels
sns.boxplot(data = numeric);
plt.ylim(0,25);
plt.figure(figsize = (10, 4));

#Rotating labels to prevent overlap
boxplot = sns.boxplot(data = numeric);
plt.ylim(0,25);
plt.setp(boxplot.get_xticklabels(), rotation=30, horizontalalignment='right');

#Displot for extreme skew in untransformed Size and Price
sns.distplot(numeric['Price'], color = 'Black', label = 'Price');
sns.distplot(numeric['Size'], color = 'Red',  label = 'Size');
plt.xlabel('Distribution of Untransformed Values')
plt.legend();

#Displot for much more symmetric LN(Size) and LN(Price)
sns.distplot(numeric['LN(Price)'], color = 'Black', label = 'LN(Price)');
sns.distplot(numeric['LN(Size)'], color = 'Red',  label = 'LN(Size)');
plt.xlabel('Distribution of Natural Log Transformation Values')
plt.legend();

#Distplot to examine skew in each of room types
sns.distplot(numeric.Bedrooms, label = 'Bedrooms');
sns.distplot(numeric.Bathrooms, label = 'Bathrooms');
sns.distplot(numeric['Car Parks'], label = 'Car Parks');
plt.xlabel('Distribution of Room Types')
plt.xticks(range(0,22,2))
plt.legend();

#REGRESSION 1: sklearn simple linear [ERROR MESSAGE VERSION]
Reg1_sklearn_x = numeric['LN(Size)']
Reg1_sklearn_y = numeric['LN(Price)']
Reg1_sklearn_model = linear_model.LinearRegression()
Reg1_sklearn_model.fit(Reg1_sklearn_x, Reg1_sklearn_y)
print('R^2= ', Reg1_sklearn_model.score(Reg1_sklearn_x, Reg1_sklearn_y))
print('Intercept:', Reg1_sklearn_model.intercept_)
print('Slope:', Reg1_sklearn_model.coef_) 

#sklearn simple linear, successful after reshaping
Reg1_sklearn_x = np.array(numeric['LN(Size)']).reshape(-1,1)
Reg1_sklearn_y = np.array(numeric['LN(Price)']).reshape(-1,1)
Reg1_sklearn_model = linear_model.LinearRegression()
Reg1_sklearn_model.fit(Reg1_sklearn_x, Reg1_sklearn_y)
print('R^2= ', Reg1_sklearn_model.score(Reg1_sklearn_x, Reg1_sklearn_y))
print('Intercept:', Reg1_sklearn_model.intercept_)
print('Slope:', Reg1_sklearn_model.coef_) 

#statsmodels simple linear
Reg1_statsmodel_x = numeric['LN(Size)']
Reg1_statsmodel_y = numeric['LN(Price)']
Reg1_statsmodel_x = sm.add_constant(Reg1_statsmodel_x)
Reg1_statsmodel = sm.OLS(Reg1_statsmodel_y, Reg1_statsmodel_x).fit() 
print(Reg1_statsmodel.summary())

#REGRESSION 2: creating and previewing a new dataframe because null values in any column would prevent multivariate regression
multivar_numeric = numeric.dropna()
multivar_numeric.info()

#sklearn multivariate linear
multivar_sklearn_x = multivar_numeric[['Bedrooms', 'Bathrooms', 'Car Parks']]
multivar_sklearn_y = multivar_numeric['LN(Size)']
multivar_sklearn_model = linear_model.LinearRegression()
multivar_sklearn_model.fit(multivar_sklearn_x, multivar_sklearn_y)
print('R^2= ', multivar_sklearn_model.score(multivar_sklearn_x, multivar_sklearn_y))
print('Intercept:', multivar_sklearn_model.intercept_)
print('Slope:', multivar_sklearn_model.coef_) 

#statsmodels multivariate linear
multivar_statsmodel_x = multivar_numeric[['Bedrooms', 'Bathrooms', 'Car Parks']]
multivar_statsmodel_y = multivar_numeric['LN(Size)']
multivar_statsmodel_x = sm.add_constant(multivar_statsmodel_x)
multivar_statsmodel = sm.OLS(multivar_statsmodel_y, multivar_statsmodel_x).fit() 
print(multivar_statsmodel.summary())

#REGRESSION 3: creating and previewing a new dataframe for which only rows with null bathroom entries must be dropped, sklearn simple linear, statsmodels simple linear
Reg3_data = numeric.dropna(subset = ['Bathrooms'])

Reg3_sklearn_x = np.array(Reg3_data['Bathrooms']).reshape(-1, 1)
Reg3_sklearn_y = np.array(Reg3_data['LN(Size)']).reshape(-1, 1)
Reg3_sklearn_model = linear_model.LinearRegression()
Reg3_sklearn_model.fit(Reg3_sklearn_x, Reg3_sklearn_y)
print('R^2= ', Reg3_sklearn_model.score(Reg3_sklearn_x, Reg3_sklearn_y))
print('Intercept:', Reg3_sklearn_model.intercept_)
print('Slope:', Reg3_sklearn_model.coef_) 

Reg3_statsmodel_x = Reg3_data['Bathrooms']
Reg3_statsmodel_y = Reg3_data['LN(Size)']
Reg3_statsmodel_x = sm.add_constant(Reg3_statsmodel_x)
Reg3_statsmodel = sm.OLS(Reg3_statsmodel_y, Reg3_statsmodel_x).fit() 
print(Reg3_statsmodel.summary())



#K-MEANS CLUSTERING
#Clustering 1: Elbow Method
interior = multivar_numeric[['Bedrooms', 'Bathrooms']]
wcss=[]
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(interior)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS');

#Visualization of scatter plot with clusters and centroids
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(interior)
plt.scatter(interior.iloc[y_kmeans==0, 0], interior.iloc[y_kmeans==0, 1], s=25, c='red', label ='Cluster 1')
plt.scatter(interior.iloc[y_kmeans==1, 0], interior.iloc[y_kmeans==1, 1], s=25, c='blue', label ='Cluster 2')
plt.scatter(interior.iloc[y_kmeans==2, 0], interior.iloc[y_kmeans==2, 1], s=25, c='green', label ='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='purple', label = 'Centroids');
plt.title("Bedroom to Bathroom Count in Malaysian Properties")
plt.ylabel("Bathrooms")
plt.yticks(range(0, 24, 4))
plt.xlabel("Bedrooms")
plt.legend();

#Clustering 2: Elbow Method
Price_to_area = numeric[['LN(Size)', 'LN(Price)']].dropna()
wcss=[]
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(Price_to_area)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS');

#Visualization of scatter plot with clusters and centroids
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(Price_to_area)
plt.scatter(Price_to_area.iloc[y_kmeans==0, 0], Price_to_area.iloc[y_kmeans==0, 1], s=25, c='red', label ='Cluster 1')
plt.scatter(Price_to_area.iloc[y_kmeans==1, 0], Price_to_area.iloc[y_kmeans==1, 1], s=25, c='blue', label ='Cluster 2')
plt.scatter(Price_to_area.iloc[y_kmeans==2, 0], Price_to_area.iloc[y_kmeans==2, 1], s=25, c='green', label ='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='purple', label = 'Centroids');
plt.title("LN(Price) and LN(Size) Combinations in Malaysian Properties")
plt.ylabel("LN(Price)")
plt.xlabel("LN(Size)")
plt.xticks(range(6, 15, 1))
plt.legend();


