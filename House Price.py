import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

HP = pd.read_csv('house_price.csv')

HP.head()

HP.shape

HP.info()

HP.isna().sum()

HP = HP.drop(HP[['Id','Alley', 'Fence', 'MiscFeature']],axis = 1)

Id :- its won't required for analysis.
Alley :- Its having 1369 Na values.
Fence :- 1179 Na's.
MiscFeature :- 1406 Na's

HP.shape

HP.head()

HP.info()

# Check for the Shape and value counts of x and y after splitting the data variables.
y = HP.iloc[:, 76]
print('Dependent_Variable: ','\n','Dimensions: ', y.shape,'\n','Column_name: ', y.name, '\n')
x = HP.iloc[:,0:76]
print('Independent_Variable: ','\n','Dimensions: ', x.shape,'\n','Column_names: ', x.columns, '\n')

x.info()

# Lets take individually based on their data types.
Num_type = ['float32', 'float64', 'int16', 'int32', 'int64']
x_num = x.select_dtypes(Num_type)
print('Total: ', len(x_num.columns))
x_num.columns

Cat_type = ['object']
x_cat = x.select_dtypes(Cat_type)
print('Total: ', len(x_cat.columns))
x_cat.columns

x_num.isnull().sum()

print(x_num['LotFrontage'].median())
print(x_num['GarageYrBlt'].median())
print(x_num['MasVnrArea'].mean())

x_num.iloc[:,1] = np.where(x_num.iloc[:,1].isnull() == True, 69, x_num.iloc[:,1])

x_num.iloc[:,24] = np.where(x_num.iloc[:,24].isnull() == True, 1980, x_num.iloc[:,24])

x_num.iloc[:,7] = np.where(x_num.iloc[:,7].isnull() == True, 103.6852, x_num.iloc[:,7])

x_num.isnull().sum()

### 1

print('Type : ',x_num.iloc[:,1].dtype)
print('Column_name : ' ,x_num.iloc[:,1].name)

print('Null_value_count: ',x_num.iloc[:,1].isna().sum())

print('Skewness: ', x_num.iloc[:,0].skew())
x_num.iloc[:,0].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,0], color = 'green')
plt.xlabel(x_num.iloc[:,0].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,0].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,0], color = 'orange')
plt.xlabel(x_num.iloc[:,0].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,0].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 2

print('Type : ',x_num.iloc[:,1].dtype)
print('Column_name : ' ,x_num.iloc[:,1].name)

print('Null_value_count: ',x_num.iloc[:,1].isna().sum())

print('Skewness: ', x_num.iloc[:,1].skew())
x_num.iloc[:,1].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,1], color = 'green')
plt.xlabel(x_num.iloc[:,1].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,1].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,1], color = 'orange')
plt.xlabel(x_num.iloc[:,1].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,1].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,1].quantile(0.05), '\n','95% :', x_num.iloc[:,1].quantile(0.95))

import numpy as np
x_num.iloc[:,1] = np.where(x_num.iloc[:,1] > x_num.iloc[:,1].quantile(0.95), x_num.iloc[:,1].quantile(0.95), x_num.iloc[:,1])
x_num.iloc[:,1].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,1], color = 'green')
plt.xlabel(x_num.iloc[:,1].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,1].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,1], color = 'orange')
plt.xlabel(x_num.iloc[:,1].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,1].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 3

print('Type : ',x_num.iloc[:,3].dtype)
print('Column_name : ' ,x_num.iloc[:,3].name)

print('Null_value_count: ',x_num.iloc[:,3].isna().sum())

print('Skewness: ', x_num.iloc[:,3].skew())
x_num.iloc[:,3].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,3], color = 'green')
plt.xlabel(x_num.iloc[:,3].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,3].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,3], color = 'orange')
plt.xlabel(x_num.iloc[:,3].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,3].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 4

print('Type : ',x_num.iloc[:,4].dtype)
print('Column_name : ' ,x_num.iloc[:,4].name)

print('Null_value_count: ',x_num.iloc[:,4].isna().sum())

print('Skewness: ', x_num.iloc[:,4].skew())
x_num.iloc[:,4].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,4], color = 'green')
plt.xlabel(x_num.iloc[:,4].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,4].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,4], color = 'orange')
plt.xlabel(x_num.iloc[:,4].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,4].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 5

print('Type : ',x_num.iloc[:,5].dtype)
print('Column_name : ' ,x_num.iloc[:,5].name)

print('Null_value_count: ',x_num.iloc[:,5].isna().sum())

print('Skewness: ', x_num.iloc[:,5].skew())
x_num.iloc[:,5].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,5], color = 'green')
plt.xlabel(x_num.iloc[:,5].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,5].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,5], color = 'orange')
plt.xlabel(x_num.iloc[:,5].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,5].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 6

print('Type : ',x_num.iloc[:,6].dtype)
print('Column_name : ' ,x_num.iloc[:,6].name)

print('Null_value_count: ',x_num.iloc[:,6].isna().sum())

print('Skewness: ', x_num.iloc[:,6].skew())
x_num.iloc[:,6].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,6], color = 'green')
plt.xlabel(x_num.iloc[:,6].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,6].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,6], color = 'orange')
plt.xlabel(x_num.iloc[:,6].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,6].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 7

print('Type : ',x_num.iloc[:,7].dtype)
print('Column_name : ' ,x_num.iloc[:,7].name)

print('Null_value_count: ',x_num.iloc[:,7].isna().sum())

print('Skewness: ', x_num.iloc[:,7].skew())
x_num.iloc[:,7].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,7], color = 'green')
plt.xlabel(x_num.iloc[:,7].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,7].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,7], color = 'orange')
plt.xlabel(x_num.iloc[:,7].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,7].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,7].quantile(0.05), '\n','95% :', x_num.iloc[:,7].quantile(0.95))

import numpy as np
x_num.iloc[:,7] = np.where(x_num.iloc[:,7] > x_num.iloc[:,7].quantile(0.95), x_num.iloc[:,7].quantile(0.95), x_num.iloc[:,7])
x_num.iloc[:,7].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,7], color = 'green')
plt.xlabel(x_num.iloc[:,7].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,7].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,7], color = 'orange')
plt.xlabel(x_num.iloc[:,7].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,7].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 8

print('Type : ',x_num.iloc[:,8].dtype)
print('Column_name : ' ,x_num.iloc[:,8].name)

print('Null_value_count: ',x_num.iloc[:,8].isna().sum())

print('Skewness: ', x_num.iloc[:,8].skew())
x_num.iloc[:,8].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,8], color = 'green')
plt.xlabel(x_num.iloc[:,8].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,8].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,8], color = 'orange')
plt.xlabel(x_num.iloc[:,8].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,8].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,8].quantile(0.05), '\n','95% :', x_num.iloc[:,8].quantile(0.95))

import numpy as np
x_num.iloc[:,8] = np.where(x_num.iloc[:,8] > x_num.iloc[:,8].quantile(0.95), x_num.iloc[:,8].quantile(0.95), x_num.iloc[:,8])
x_num.iloc[:,8].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,8], color = 'green')
plt.xlabel(x_num.iloc[:,8].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,8].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,8], color = 'orange')
plt.xlabel(x_num.iloc[:,8].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,8].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 9

print('Type : ',x_num.iloc[:,9].dtype)
print('Column_name : ' ,x_num.iloc[:,9].name)

print('Null_value_count: ',x_num.iloc[:,9].isna().sum())

print('Skewness: ', x_num.iloc[:,9].skew())
x_num.iloc[:,9].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,9], color = 'green')
plt.xlabel(x_num.iloc[:,9].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,9].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,9], color = 'orange')
plt.xlabel(x_num.iloc[:,9].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,9].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,9].quantile(0.05), '\n','99% :', x_num.iloc[:,9].quantile(0.99))

import numpy as np
x_num.iloc[:,9] = np.where(x_num.iloc[:,9] > x_num.iloc[:,9].quantile(0.99), x_num.iloc[:,9].quantile(0.99), x_num.iloc[:,9])
x_num.iloc[:,9].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,9], color = 'green')
plt.xlabel(x_num.iloc[:,9].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,9].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,9], color = 'orange')
plt.xlabel(x_num.iloc[:,9].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,9].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 10

print('Type : ',x_num.iloc[:,10].dtype)
print('Column_name : ' ,x_num.iloc[:,10].name)

print('Null_value_count: ',x_num.iloc[:,10].isna().sum())

print('Skewness: ', x_num.iloc[:,10].skew())
x_num.iloc[:,10].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,10], color = 'green')
plt.xlabel(x_num.iloc[:,10].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,10].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,10], color = 'orange')
plt.xlabel(x_num.iloc[:,10].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,10].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 11

print('Type : ',x_num.iloc[:,11].dtype)
print('Column_name : ' ,x_num.iloc[:,11].name)

print('Null_value_count: ',x_num.iloc[:,11].isna().sum())

print('Skewness: ', x_num.iloc[:,11].skew())
x_num.iloc[:,11].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,11], color = 'green')
plt.xlabel(x_num.iloc[:,11].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,11].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,11], color = 'orange')
plt.xlabel(x_num.iloc[:,11].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,11].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,11].quantile(0.05), '\n','95% :', x_num.iloc[:,11].quantile(0.95))

import numpy as np
x_num.iloc[:,11] = np.where(x_num.iloc[:,11] > x_num.iloc[:,11].quantile(0.95), x_num.iloc[:,11].quantile(0.95), x_num.iloc[:,11])
x_num.iloc[:,11].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,11], color = 'green')
plt.xlabel(x_num.iloc[:,11].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,11].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,11], color = 'orange')
plt.xlabel(x_num.iloc[:,11].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,11].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 12

print('Type : ',x_num.iloc[:,12].dtype)
print('Column_name : ' ,x_num.iloc[:,12].name)

print('Null_value_count: ',x_num.iloc[:,12].isna().sum())

print('Skewness: ', x_num.iloc[:,12].skew())
x_num.iloc[:,12].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,12], color = 'green')
plt.xlabel(x_num.iloc[:,12].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,12].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,12], color = 'orange')
plt.xlabel(x_num.iloc[:,12].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,12].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,12].quantile(0.05), '\n','95% :', x_num.iloc[:,12].quantile(0.95))

import numpy as np
x_num.iloc[:,12] = np.where(x_num.iloc[:,12] > x_num.iloc[:,12].quantile(0.95), x_num.iloc[:,12].quantile(0.95), x_num.iloc[:,12])
x_num.iloc[:,12].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,12], color = 'green')
plt.xlabel(x_num.iloc[:,12].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,12].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,12], color = 'orange')
plt.xlabel(x_num.iloc[:,12].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,12].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 13

print('Type : ',x_num.iloc[:,13].dtype)
print('Column_name : ' ,x_num.iloc[:,13].name)

print('Null_value_count: ',x_num.iloc[:,13].isna().sum())

print('Skewness: ', x_num.iloc[:,13].skew())
x_num.iloc[:,13].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,13], color = 'green')
plt.xlabel(x_num.iloc[:,13].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,13].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,13], color = 'orange')
plt.xlabel(x_num.iloc[:,13].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,13].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 14

print('Type : ',x_num.iloc[:,14].dtype)
print('Column_name : ' ,x_num.iloc[:,14].name)

print('Null_value_count: ',x_num.iloc[:,14].isna().sum())

print('Skewness: ', x_num.iloc[:,14].skew())
x_num.iloc[:,14].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,14], color = 'green')
plt.xlabel(x_num.iloc[:,14].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,14].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,14], color = 'orange')
plt.xlabel(x_num.iloc[:,14].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,14].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,14].quantile(0.05), '\n','99% :', x_num.iloc[:,14].quantile(0.99))

import numpy as np
x_num.iloc[:,14] = np.where(x_num.iloc[:,14] > x_num.iloc[:,14].quantile(0.99), x_num.iloc[:,14].quantile(0.99), x_num.iloc[:,14])
x_num.iloc[:,14].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,14], color = 'green')
plt.xlabel(x_num.iloc[:,14].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,14].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,14], color = 'orange')
plt.xlabel(x_num.iloc[:,14].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,14].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 15

print('Type : ',x_num.iloc[:,15].dtype)
print('Column_name : ' ,x_num.iloc[:,15].name)

print('Null_value_count: ',x_num.iloc[:,15].isna().sum())

print('Skewness: ', x_num.iloc[:,15].skew())
x_num.iloc[:,15].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,15], color = 'green')
plt.xlabel(x_num.iloc[:,15].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,15].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,15], color = 'orange')
plt.xlabel(x_num.iloc[:,15].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,15].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,15].quantile(0.05), '\n','95% :', x_num.iloc[:,15].quantile(0.95))

import numpy as np
x_num.iloc[:,15] = np.where(x_num.iloc[:,15] > x_num.iloc[:,15].quantile(0.95), x_num.iloc[:,15].quantile(0.95), x_num.iloc[:,15])
x_num.iloc[:,15].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,15], color = 'green')
plt.xlabel(x_num.iloc[:,15].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,15].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,15], color = 'orange')
plt.xlabel(x_num.iloc[:,15].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,15].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 16

print('Type : ',x_num.iloc[:,16].dtype)
print('Column_name : ' ,x_num.iloc[:,16].name)

print('Null_value_count: ',x_num.iloc[:,16].isna().sum())

print('Skewness: ', x_num.iloc[:,16].skew())
x_num.iloc[:,16].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,16], color = 'green')
plt.xlabel(x_num.iloc[:,16].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,16].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,16], color = 'orange')
plt.xlabel(x_num.iloc[:,16].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,16].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 17

print('Type : ',x_num.iloc[:,17].dtype)
print('Column_name : ' ,x_num.iloc[:,17].name)

print('Null_value_count: ',x_num.iloc[:,17].isna().sum())

print('Skewness: ', x_num.iloc[:,17].skew())
x_num.iloc[:,17].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,17], color = 'green')
plt.xlabel(x_num.iloc[:,17].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,17].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,17], color = 'orange')
plt.xlabel(x_num.iloc[:,17].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,17].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 18

print('Type : ',x_num.iloc[:,18].dtype)
print('Column_name : ' ,x_num.iloc[:,18].name)

print('Null_value_count: ',x_num.iloc[:,18].isna().sum())

print('Skewness: ', x_num.iloc[:,18].skew())
x_num.iloc[:,18].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,18], color = 'green')
plt.xlabel(x_num.iloc[:,18].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,18].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,18], color = 'orange')
plt.xlabel(x_num.iloc[:,18].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,18].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 19

print('Type : ',x_num.iloc[:,19].dtype)
print('Column_name : ' ,x_num.iloc[:,19].name)

print('Null_value_count: ',x_num.iloc[:,19].isna().sum())

print('Skewness: ', x_num.iloc[:,19].skew())
x_num.iloc[:,19].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,19], color = 'green')
plt.xlabel(x_num.iloc[:,19].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,19].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,19], color = 'orange')
plt.xlabel(x_num.iloc[:,19].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,19].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 20

print('Type : ',x_num.iloc[:,20].dtype)
print('Column_name : ' ,x_num.iloc[:,20].name)

print('Null_value_count: ',x_num.iloc[:,20].isna().sum())

print('Skewness: ', x_num.iloc[:,20].skew())
x_num.iloc[:,20].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,20], color = 'green')
plt.xlabel(x_num.iloc[:,20].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,20].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,20], color = 'orange')
plt.xlabel(x_num.iloc[:,20].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,20].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 21

print('Type : ',x_num.iloc[:,21].dtype)
print('Column_name : ' ,x_num.iloc[:,21].name)

print('Null_value_count: ',x_num.iloc[:,21].isna().sum())

print('Skewness: ', x_num.iloc[:,21].skew())
x_num.iloc[:,21].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,21], color = 'green')
plt.xlabel(x_num.iloc[:,21].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,21].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,21], color = 'orange')
plt.xlabel(x_num.iloc[:,21].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,21].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 22

print('Type : ',x_num.iloc[:,22].dtype)
print('Column_name : ' ,x_num.iloc[:,22].name)

print('Null_value_count: ',x_num.iloc[:,22].isna().sum())

print('Skewness: ', x_num.iloc[:,22].skew())
x_num.iloc[:,22].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,22], color = 'green')
plt.xlabel(x_num.iloc[:,22].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,22].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,22], color = 'orange')
plt.xlabel(x_num.iloc[:,22].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,22].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 23

print('Type : ',x_num.iloc[:,23].dtype)
print('Column_name : ' ,x_num.iloc[:,23].name)

print('Null_value_count: ',x_num.iloc[:,23].isna().sum())

print('Skewness: ', x_num.iloc[:,23].skew())
x_num.iloc[:,23].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,23], color = 'green')
plt.xlabel(x_num.iloc[:,23].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,23].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,23], color = 'orange')
plt.xlabel(x_num.iloc[:,23].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,23].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 24

print('Type : ',x_num.iloc[:,24].dtype)
print('Column_name : ' ,x_num.iloc[:,24].name)

print('Null_value_count: ',x_num.iloc[:,24].isna().sum())

print('Skewness: ', x_num.iloc[:,24].skew())
x_num.iloc[:,24].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,24], color = 'green')
plt.xlabel(x_num.iloc[:,24].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,24].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,24], color = 'orange')
plt.xlabel(x_num.iloc[:,24].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,24].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 25

print('Type : ',x_num.iloc[:,25].dtype)
print('Column_name : ' ,x_num.iloc[:,25].name)

print('Null_value_count: ',x_num.iloc[:,25].isna().sum())

print('Skewness: ', x_num.iloc[:,25].skew())
x_num.iloc[:,25].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,25], color = 'green')
plt.xlabel(x_num.iloc[:,25].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,25].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,25], color = 'orange')
plt.xlabel(x_num.iloc[:,25].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,25].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 26

print('Type : ',x_num.iloc[:,26].dtype)
print('Column_name : ' ,x_num.iloc[:,26].name)

print('Null_value_count: ',x_num.iloc[:,26].isna().sum())

print('Skewness: ', x_num.iloc[:,26].skew())
x_num.iloc[:,26].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,26], color = 'green')
plt.xlabel(x_num.iloc[:,26].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,26].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,26], color = 'orange')
plt.xlabel(x_num.iloc[:,26].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,26].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 27

print('Type : ',x_num.iloc[:,27].dtype)
print('Column_name : ' ,x_num.iloc[:,27].name)

print('Null_value_count: ',x_num.iloc[:,27].isna().sum())

print('Skewness: ', x_num.iloc[:,27].skew())
x_num.iloc[:,27].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,27], color = 'green')
plt.xlabel(x_num.iloc[:,27].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,27].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,27], color = 'orange')
plt.xlabel(x_num.iloc[:,27].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,27].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 28

print('Type : ',x_num.iloc[:,28].dtype)
print('Column_name : ' ,x_num.iloc[:,28].name)

print('Null_value_count: ',x_num.iloc[:,28].isna().sum())

print('Skewness: ', x_num.iloc[:,28].skew())
x_num.iloc[:,28].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,28], color = 'green')
plt.xlabel(x_num.iloc[:,28].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,28].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,28], color = 'orange')
plt.xlabel(x_num.iloc[:,28].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,28].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,28].quantile(0.05), '\n','99% :', x_num.iloc[:,28].quantile(0.99))

import numpy as np
x_num.iloc[:,28] = np.where(x_num.iloc[:,28] > x_num.iloc[:,28].quantile(0.99), x_num.iloc[:,28].quantile(0.99), x_num.iloc[:,28])
x_num.iloc[:,28].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,28], color = 'green')
plt.xlabel(x_num.iloc[:,28].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,28].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,28], color = 'orange')
plt.xlabel(x_num.iloc[:,28].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,28].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 29

print('Type : ',x_num.iloc[:,29].dtype)
print('Column_name : ' ,x_num.iloc[:,29].name)

print('Null_value_count: ',x_num.iloc[:,29].isna().sum())

print('Skewness: ', x_num.iloc[:,29].skew())
x_num.iloc[:,29].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,29], color = 'green')
plt.xlabel(x_num.iloc[:,29].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,29].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,29], color = 'orange')
plt.xlabel(x_num.iloc[:,29].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,29].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,29].quantile(0.05), '\n','99% :', x_num.iloc[:,29].quantile(0.999))

import numpy as np
x_num.iloc[:,29] = np.where(x_num.iloc[:,29] > x_num.iloc[:,29].quantile(0.999), x_num.iloc[:,29].quantile(0.999), x_num.iloc[:,29])
x_num.iloc[:,29].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,29], color = 'green')
plt.xlabel(x_num.iloc[:,29].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,29].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,29], color = 'orange')
plt.xlabel(x_num.iloc[:,29].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,29].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 30

print('Type : ',x_num.iloc[:,30].dtype)
print('Column_name : ' ,x_num.iloc[:,30].name)

print('Null_value_count: ',x_num.iloc[:,30].isna().sum())

print('Skewness: ', x_num.iloc[:,30].skew())
x_num.iloc[:,30].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,30], color = 'green')
plt.xlabel(x_num.iloc[:,30].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,30].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,30], color = 'orange')
plt.xlabel(x_num.iloc[:,30].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,30].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,30].quantile(0.05), '\n','99% :', x_num.iloc[:,30].quantile(0.999))

import numpy as np
x_num.iloc[:,30] = np.where(x_num.iloc[:,30] > x_num.iloc[:,30].quantile(0.999), x_num.iloc[:,30].quantile(0.999), x_num.iloc[:,30])
x_num.iloc[:,30].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,30], color = 'green')
plt.xlabel(x_num.iloc[:,30].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,30].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,30], color = 'orange')
plt.xlabel(x_num.iloc[:,30].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,30].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 31

print('Type : ',x_num.iloc[:,31].dtype)
print('Column_name : ' ,x_num.iloc[:,31].name)

print('Null_value_count: ',x_num.iloc[:,31].isna().sum())

print('Skewness: ', x_num.iloc[:,31].skew())
x_num.iloc[:,31].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,31], color = 'green')
plt.xlabel(x_num.iloc[:,31].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,31].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,31], color = 'orange')
plt.xlabel(x_num.iloc[:,31].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,31].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,31].quantile(0.05), '\n','99% :', x_num.iloc[:,31].quantile(0.999))

import numpy as np
x_num.iloc[:,31] = np.where(x_num.iloc[:,31] > x_num.iloc[:,31].quantile(0.999), x_num.iloc[:,31].quantile(0.999), x_num.iloc[:,31])
x_num.iloc[:,31].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,31], color = 'green')
plt.xlabel(x_num.iloc[:,31].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,31].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,31], color = 'orange')
plt.xlabel(x_num.iloc[:,31].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,31].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 32

print('Type : ',x_num.iloc[:,32].dtype)
print('Column_name : ' ,x_num.iloc[:,32].name)

print('Null_value_count: ',x_num.iloc[:,32].isna().sum())

print('Skewness: ', x_num.iloc[:,32].skew())
x_num.iloc[:,32].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,32], color = 'green')
plt.xlabel(x_num.iloc[:,32].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,32].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,32], color = 'orange')
plt.xlabel(x_num.iloc[:,32].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,32].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 33

print('Type : ',x_num.iloc[:,33].dtype)
print('Column_name : ' ,x_num.iloc[:,33].name)

print('Null_value_count: ',x_num.iloc[:,33].isna().sum())

print('Skewness: ', x_num.iloc[:,33].skew())
x_num.iloc[:,33].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,33], color = 'green')
plt.xlabel(x_num.iloc[:,33].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,33].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,33], color = 'orange')
plt.xlabel(x_num.iloc[:,33].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,33].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('5% :', x_num.iloc[:,33].quantile(0.05), '\n','98% :', x_num.iloc[:,33].quantile(0.98))

import numpy as np
x_num.iloc[:,33] = np.where(x_num.iloc[:,33] > x_num.iloc[:,33].quantile(0.98), x_num.iloc[:,33].quantile(0.98), x_num.iloc[:,33])
x_num.iloc[:,33].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,33], color = 'green')
plt.xlabel(x_num.iloc[:,33].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,33].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,33], color = 'orange')
plt.xlabel(x_num.iloc[:,33].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,33].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 34

print('Type : ',x_num.iloc[:,34].dtype)
print('Column_name : ' ,x_num.iloc[:,34].name)

print('Null_value_count: ',x_num.iloc[:,34].isna().sum())

print('Skewness: ', x_num.iloc[:,34].skew())
x_num.iloc[:,34].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,34], color = 'green')
plt.xlabel(x_num.iloc[:,34].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,34].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,34], color = 'orange')
plt.xlabel(x_num.iloc[:,34].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,34].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

### 35

print('Type : ',x_num.iloc[:,35].dtype)
print('Column_name : ' ,x_num.iloc[:,35].name)

print('Null_value_count: ',x_num.iloc[:,35].isna().sum())

print('Skewness: ', x_num.iloc[:,35].skew())
x_num.iloc[:,35].describe()

plt.subplot(1,2,1)
sns.boxplot(x_num.iloc[:,35], color = 'green')
plt.xlabel(x_num.iloc[:,35].name, fontsize = 20)
plt.title('Boxplot_ '+ x_num.iloc[:,35].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(x_num.iloc[:,35], color = 'orange')
plt.xlabel(x_num.iloc[:,35].name, fontsize = 20)
plt.title('Histogram_ '+ x_num.iloc[:,35].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

v = x_num

# Mean-Max Normalization
x_num = (x_num - x_num.min()) / (x_num.max() - x_num.min())

x_num.head()

x_cat.head()

x_cat.isnull().sum()

x_cat.drop(['PoolQC','MasVnrType'], axis = 1, inplace = True)

x_cat.info()

x_cat.columns

x_cat.iloc[:,19] = np.where(x_cat.iloc[:,19].isnull() == True, x_cat.iloc[:,19].mode(), x_cat.iloc[:,19])
x_cat.iloc[:,20] = np.where(x_cat.iloc[:,20].isnull() == True, x_cat.iloc[:,20].mode(), x_cat.iloc[:,20])
x_cat.iloc[:,21] = np.where(x_cat.iloc[:,21].isnull() == True, x_cat.iloc[:,21].mode(), x_cat.iloc[:,21])
x_cat.iloc[:,22] = np.where(x_cat.iloc[:,22].isnull() == True, x_cat.iloc[:,22].mode(), x_cat.iloc[:,22])
x_cat.iloc[:,23] = np.where(x_cat.iloc[:,23].isnull() == True, x_cat.iloc[:,23].mode(), x_cat.iloc[:,23])
x_cat.iloc[:,27] = np.where(x_cat.iloc[:,27].isnull() == True, x_cat.iloc[:,27].mode(), x_cat.iloc[:,27])
x_cat.iloc[:,30] = np.where(x_cat.iloc[:,30].isnull() == True, x_cat.iloc[:,30].mode(), x_cat.iloc[:,30])
x_cat.iloc[:,31] = np.where(x_cat.iloc[:,31].isnull() == True, x_cat.iloc[:,31].mode(), x_cat.iloc[:,31])
x_cat.iloc[:,32] = np.where(x_cat.iloc[:,32].isnull() == True, x_cat.iloc[:,32].mode(), x_cat.iloc[:,32])
x_cat.iloc[:,33] = np.where(x_cat.iloc[:,33].isnull() == True, x_cat.iloc[:,33].mode(), x_cat.iloc[:,33])
x_cat.iloc[:,34] = np.where(x_cat.iloc[:,34].isnull() == True, x_cat.iloc[:,34].mode(), x_cat.iloc[:,34])

x_cat.isnull().sum()

x_cat = pd.DataFrame(x_cat)

x_cat.head()

c = x_cat

from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
x_cat = pd.DataFrame(ord_enc.fit_transform(x_cat))

x_cat.rename(columns = {0 : 'MSZoning', 1 : 'Street', 2 : 'LotShape', 3 : 'LandContour', 4 : 'Utilities', 5 : 'LotConfig', 6 : 'LandSlope', 7 : 'Neighborhood', 8 : 'Condition1', 
       9 : 'Condition2', 10 : 'BldgType', 11 : 'HouseStyle', 12 : 'RoofStyle', 13 : 'RoofMatl', 14 : 'Exterior1st', 15 : 'Exterior2nd', 16 : 'ExterQual', 17 : 'ExterCond', 
       18 : 'Foundation', 19 : 'BsmtQual', 20 : 'BsmtCond', 21 : 'BsmtExposure', 22 : 'BsmtFinType1', 23 : 'BsmtFinType2', 24 : 'Heating', 25 : 'HeatingQC', 26 : 'CentralAir', 
       27 : 'Electrical', 28 : 'KitchenQual', 29 : 'Functional', 30 : 'FireplaceQu', 31 : 'GarageType', 32 : 'GarageFinish', 33 : 'GarageQual', 34 : 'GarageCond', 35 : 'PavedDrive', 36 : 'SaleType',
       37 : 'SaleCondition'}, inplace = True)

x_cat.head()

x_cat.isnull().sum()

df = pd.concat([x_num, x_cat], axis = 1)

df.head()

### `Train and Test Data Splitting`

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, y, train_size = 0.8, random_state = 100)

print('x_train shape :', x_train.shape)
print('x_test shape :', x_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)

### `Linear Regression`

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                               # Y-label

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 

y_pred

### `Check for R^2 and MSE and VIF`

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)

### `LOGIT OLS SUMMARY`

import statsmodels.api as sm
x_train_sm = x_train
x_train_sm = sm.add_constant(x_train_sm)
lr = sm.OLS(y_train,x_train_sm).fit()

lr.params
print(lr.summary())

### RFE

from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(x_train, y_train)
#print(selector.support_)
print(selector.ranking_)

### `1st Model`

x1_train=x_train[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                  'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', '3SsnPorch', 'MiscVal',
                  'MoSold', 'YrSold', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                  'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                  'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']] 
x1_test =x_test[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                  'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', '3SsnPorch', 'MiscVal',
                  'MoSold', 'YrSold', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                  'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                  'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']] 

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr1 = lr.fit(x1_train,y_train)

y_pred = lr1.predict(x1_test)

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                               # Y-label

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 

x1_train_sm = sm.add_constant(x1_train)
lr_1 = sm.OLS(y_train,x1_train_sm).fit()

lr_1.params
print(lr_1.summary())

### ` 2nd Model`

x2_train=x_train[['LotArea', 'OverallQual', 'OverallCond', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                  'GarageCars', 'Street', 'LotShape', 'LandContour', 'Condition2', 'RoofMatl', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleCondition']]
x2_test =x_test[['LotArea', 'OverallQual', 'OverallCond', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                 'GarageCars', 'Street', 'LotShape', 'LandContour', 'Condition2', 'RoofMatl', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleCondition']]

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x2_train.columns

vif_data["VIF"] = [variance_inflation_factor(x2_train.values, i) 
                          for i in range(len(x2_train.columns))] 
  
print(vif_data)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr2 = lr.fit(x2_train,y_train)

y_pred = lr2.predict(x2_test)

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                               # Y-label

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 

x2_train_sm = sm.add_constant(x2_train)
lr_2 = sm.OLS(y_train,x2_train_sm).fit()

lr_2.params
print(lr_2.summary())

### `3rd Model`

x3_train=x_train[['LotArea', 'OverallQual', 'OverallCond', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                  'GarageCars', 'Street', 'LotShape', 'LandContour', 'RoofMatl', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleCondition']]
x3_test =x_test[['LotArea', 'OverallQual', 'OverallCond', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                  'GarageCars', 'Street', 'LotShape', 'LandContour', 'RoofMatl', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleCondition']]

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x3_train.columns

vif_data["VIF"] = [variance_inflation_factor(x3_train.values, i) 
                          for i in range(len(x3_train.columns))] 
  
print(vif_data)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr3 = lr.fit(x3_train,y_train)

y_pred = lr3.predict(x3_test)

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                               # Y-label

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 

import statsmodels.api as sm
x3_train_sm = sm.add_constant(x3_train)
lr_3 = sm.OLS(y_train,x3_train_sm).fit()

lr_3.params
print(lr_3.summary())

### `4th Model`

x4_train=x_train[['OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1',  'GrLivArea', 'KitchenAbvGr', 'Fireplaces',
                  'GarageCars', 'Street', 'LotShape', 'LandContour', 'RoofMatl', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleCondition']]
x4_test =x_test[['OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFinSF1', 'KitchenAbvGr', 'Fireplaces',
                  'GarageCars', 'Street', 'LotShape', 'LandContour', 'RoofMatl', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleCondition']]

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x4_train.columns

vif_data["VIF"] = [variance_inflation_factor(x4_train.values, i) 
                          for i in range(len(x4_train.columns))] 
  
print(vif_data)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr4 = lr.fit(x4_train,y_train)

y_pred = lr4.predict(x4_test)

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                               # Y-label

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 

import statsmodels.api as sm
x4_train_sm = sm.add_constant(x4_train)
lr_4 = sm.OLS(y_train,x4_train_sm).fit()

lr_4.params
print(lr_4.summary())

### `5th Model`

x5_train=x_train[['OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'KitchenAbvGr', 'Fireplaces',
                  'GarageCars', 'LotShape', 'LandContour', 'RoofMatl', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleCondition']]
x5_test =x_test[['OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'KitchenAbvGr', 'Fireplaces',
                  'GarageCars', 'LotShape', 'LandContour', 'RoofMatl', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleCondition']]

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x5_train.columns

vif_data["VIF"] = [variance_inflation_factor(x5_train.values, i) 
                          for i in range(len(x5_train.columns))] 
  
print(vif_data)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr5 = lr.fit(x5_train,y_train)

y_pred = lr5.predict(x5_test)

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                               # Y-label

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 

import statsmodels.api as sm
x5_train_sm = sm.add_constant(x5_train)
lr_5 = sm.OLS(y_train,x5_train_sm).fit()

lr_5.params
print(lr_5.summary())

### `6th Model`

x6_train=x_train[['OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'KitchenAbvGr',
                  'GarageCars', 'LandContour', 'RoofMatl', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleCondition']]
x6_test =x_test[['OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'KitchenAbvGr',
                  'GarageCars', 'LandContour', 'RoofMatl', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'KitchenQual', 'Functional', 'SaleCondition']]

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x6_train.columns

vif_data["VIF"] = [variance_inflation_factor(x6_train.values, i) 
                          for i in range(len(x6_train.columns))] 
  
print(vif_data)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr6 = lr.fit(x6_train,y_train)

y_pred = lr6.predict(x6_test)

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                               # Y-label

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 

import statsmodels.api as sm
x6_train_sm = sm.add_constant(x6_train)
lr_6 = sm.OLS(y_train,x6_train_sm).fit()

lr_6.params
print(lr_6.summary())

### `7th Model`

x7_train=x_train[['OverallQual','OverallCond', '1stFlrSF', 'BsmtFinSF1', 'KitchenAbvGr', '2ndFlrSF',
                  'BsmtQual', 'KitchenQual']]
x7_test =x_test[['OverallQual', 'OverallCond', '1stFlrSF', 'BsmtFinSF1', 'KitchenAbvGr', '2ndFlrSF',
                  'BsmtQual', 'KitchenQual']]

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x7_train.columns

vif_data["VIF"] = [variance_inflation_factor(x7_train.values, i) 
                          for i in range(len(x7_train.columns))] 
  
print(vif_data)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr7 = lr.fit(x7_train,y_train)

y_pred = lr7.predict(x7_test)

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                               # Y-label

c = [i for i in range(1,293,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Error_line', fontsize=16) 

import statsmodels.api as sm
x7_train_sm = sm.add_constant(x7_train)
lr_7 = sm.OLS(y_train,x7_train_sm).fit()

lr_7.params
print(lr_7.summary())

q = pd.DataFrame(df[['OverallQual','OverallCond', '1stFlrSF', 'BsmtFinSF1', 'KitchenAbvGr', '2ndFlrSF',
                  'BsmtQual', 'KitchenQual']])
y = pd.DataFrame(y)

dff = pd.concat([q,y], axis = 1)

corelation_table = dff.corr()
corelation_table.to_csv(r'corelation_table.csv', index = False)

plt.figure(figsize = (15,10))
sns.heatmap(dff.corr(),annot = True)
plt.show()

### `VISUALIZATION :`

q = pd.DataFrame(df[['OverallQual','OverallCond', '1stFlrSF', 'BsmtFinSF1', 'KitchenAbvGr', '2ndFlrSF',
                  'BsmtQual', 'KitchenQual']])
y = pd.DataFrame(y)

dff = pd.concat([q,y], axis = 1)

sns.distplot(HP['SalePrice'])

sns.catplot(x='KitchenQual', y='SalePrice', hue='OverallQual' ,data= HP, kind='bar')

sns.catplot(x='OverallQual', y='SalePrice', hue='OverallCond' ,data= HP, kind='bar')

sns.catplot(x='KitchenQual', y='SalePrice', hue='KitchenAbvGr' ,data= HP, kind='bar')

sns.catplot(x='KitchenAbvGr', y='SalePrice', hue='OverallQual' ,data= HP, kind='bar')

plt.scatter(HP['OverallQual'], HP['SalePrice'], c = HP['KitchenAbvGr'])
plt.legend(HP['KitchenQual'])
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

lines = plt.scatter(HP['KitchenQual'], HP['SalePrice'])
plt.xlabel('KitchenQual')
plt.ylabel('SalePrice')
plt.setp(lines, 'color', 'r', 'linewidth', 2.0)
plt.show()

#### `Correlation-Plot`

corr = dff.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

sns.pairplot(dff)
plt.show()

### `K-MEANS`

#### `Select samples from the data randomly`

sample = dff.sample(frac = 0.30, replace = False, random_state = 100)
print(len(sample))
sample

#### `Scaling`

from sklearn.preprocessing import StandardScaler
data_scaled = StandardScaler().fit_transform(dff)
data_scaled

#### `Optimization Plot with Elbow Method`

from sklearn.cluster import KMeans
plt.figure(figsize = (10,3))
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'random', random_state = 42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

print(wcss)

plt.plot(range(1,11), wcss, 'bx-')
plt.title('The Elbow Method')
plt.axvline(4, color = 'red', linestyle = '--')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()

#### `Cluster Membership`

kmeans = KMeans(n_clusters = 4) # just making the cluster in the backned (not fitted to dataset here)
clusters = kmeans.fit_predict(data_scaled)
clusters

#### `Add Cluster Column to the actual data`

# Let's add column 'cluster' to the data

Final_cluster = clusters + 1
Cluster = list(Final_cluster)
dff['cluster'] = Cluster
dff.head()

#### `Cluster - 1`

dff[dff['cluster'] == 1]

#### `Cluster - 2`

dff[dff['cluster'] == 2]

#### `Cluster - 3`

dff[dff['cluster'] == 3]

#### `Cluster - 4`

dff[dff['cluster'] == 4]

#### `Cluster Profiling`

dff.groupby('cluster').mean()

#### `Cluster Plot`

plt.figure(figsize = (12,6))
sns.scatterplot(dff['OverallQual'], dff['SalePrice'], hue = Final_cluster, palette = ['green', 'orange', 'blue', 'red'])

#### `Overall Quality and SalesPrice`

plt.plot(dff['OverallQual']*100, 'r^', dff['SalePrice']/10000, 'g^')
plt.xlabel('Index')
plt.ylabel('Overall Quality*100 & Sales Price/10000')
plt.show()

## `PCA`

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import scale
import seaborn as sns
from sklearn.decomposition import PCA
%matplotlib inline

y.head()

df1 = df[['OverallQual','OverallCond', '1stFlrSF', 'BsmtFinSF1', 'KitchenAbvGr', '2ndFlrSF',
                  'BsmtQual', 'KitchenQual']]

df1.to_csv('df1.csv')

#convert it to numpy arrays
X=df1.values
#Scaling the values
X = scale(X)

X = pd.DataFrame(X)
X.to_csv('scal.csv')

pca = PCA(n_components=8)
pca.fit(X)

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_
var

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1

plt.plot(var1)

#Looking at above plot I'm taking 4 variables
pca = PCA(n_components=5)
pca.fit(X)
X1=pca.fit_transform(X)
print(X1)

X1 = pd.DataFrame(X1)
X1.to_csv('X1.csv')

### `PCA Model`

xp_train=x_train[['OverallQual', '1stFlrSF', 'BsmtFinSF1',
                  'BsmtQual', 'KitchenQual']]
xp_test =x_test[['OverallQual', '1stFlrSF', 'BsmtFinSF1',
                  'BsmtQual', 'KitchenQual']]

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = xp_train.columns

vif_data["VIF"] = [variance_inflation_factor(xp_train.values, i) 
                          for i in range(len(xp_train.columns))] 
  
print(vif_data)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lrp = lr.fit(xp_train,y_train)

y_pred = lrp.predict(xp_test)

import statsmodels.api as sm
xp_train_sm = sm.add_constant(xp_train)
lr_p = sm.OLS(y_train,xp_train_sm).fit()

lr_p.params
print(lr_p.summary())

q = pd.DataFrame(df[['OverallQual', '1stFlrSF', 'BsmtFinSF1', 'BsmtQual', 'KitchenQual']])
y = pd.DataFrame(y)

dff = pd.concat([q,y], axis = 1)

corelation_table = dff.corr()

plt.figure(figsize = (15,10))
sns.heatmap(dff.corr(),annot = True)
plt.show()