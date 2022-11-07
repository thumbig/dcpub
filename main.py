
###############################################################
# Data Validation
# Check all variables in the data against the criteria in the dataset above

import pandas as pd
import numpy as np
df = pd.read_csv("data/fitness_class.csv")
print("df.shape:")
print(df.shape)

print("df.info():")
df.info()

# This shows that there are no missing values in any column.
print("df.isna().sum():", df.isna().sum())

# The range and quantiles of the numeric variables are consistent with what is expectable from the context domain:
df.describe()

# Let's check whether initially categorical variables fall within sensible discreet values:
print(df['day_of_week'].value_counts())
print(df['time'].value_counts())
print(df['class_category'].value_counts())
print(df['class_capacity'].value_counts())

#Transform:
# Convert day_of_week into an ordinal variable.
# This assumes that if anything is significant, it is when in the week events occur, and not whether events occur on any particular day of the week.
# Looking at the histogram, it isn't evident that either assumption is preferable.
df['after_monday'] = df['day_of_week'].replace({'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6 })
df['time_binary'] = df['time'] == 'PM'
print(df['after_monday'].value_counts())
print(df['time_binary'].value_counts())
df.drop('day_of_week', axis=1, inplace=True)
df.drop('time', axis=1, inplace=True)

# class_capacity should be converted into a binary variable for small and large classes given that most cases fall into 2 unique values
df['capacity_binary'] = df['class_capacity'] > 20
print(df['capacity_binary'].value_counts())
df.drop('class_capacity', axis=1, inplace=True)

# over_6_month, new_students might be correlated with attendance since number of students counted in each are also likely to actully attend and be counted in the attendance column
# We'll convert these to proportions to fix this
# The resulting features are just proxies rather than measurements of anything real because  over_6_month and new_students measure students who signed up
# whereas attendance counts students who actually attended
df['over_6_month_ratio'] = df['over_6_month'] / df['attendance']
df['new_students_ratio'] = df['new_students'] / df['attendance']

# ### Possible dependent variables:
# - attendance
#
# ### Possible numeric independent variables:
# - class_capacity
# - new_students
# - over_6_month
# - age
# - days_before
#
# ### Categoric variables:
# - day_of_week
# - class_category
# - time (AM/PM)

raw_columns = ['age', 'attendance', 'days_before', 'over_6_month', 'new_students']
distributed_columns = ['age', 'attendance', 'days_before', 'over_6_month_ratio', 'new_students_ratio']
model_columns = ['age', 'after_monday', 'attendance', 'days_before', 'over_6_month_ratio', 'new_students_ratio', 'time_binary', 'capacity_binary']

###############################################################
## Exploratory Analysis
# Explore the characteristics of the variables in the data

###############################################################
# Data visualization
# The data visualization charts are shown when the ipython workspace execuated from beginning to end.

# Looking at the histograms:
# these variables appear more normally distributed:
# these variables appear less normally distributed:


import matplotlib.pyplot as plt
import seaborn as sns
SEED=99

df[model_columns].hist(alpha=0.5, figsize=(20, 10))
plt.tight_layout()
plt.savefig('hist.png')
plt.close()
# Days before might be exponentially distributed

sns.pairplot(df[distributed_columns])
plt.savefig('pairplot.png')
plt.close()

# todo: also try this after scaling
df[raw_columns].boxplot(rot=30)
plt.tight_layout()
plt.savefig('boxplot.png')
plt.close()

g = sns.scatterplot(data=df, x='age', y='attendance', alpha=0.5)
plt.xticks(np.arange(5, 55, 5))
plt.savefig('scatterplot.png')
plt.close()

sns.violinplot(data=df, x='class_category', y='attendance', random_state=SEED)
plt.savefig('violinplot.png')
plt.close()

## Observations about feature distributions in exporatory analysis:

# The violin plots show that the distributions of attendence for each class category are similar and sightly assymetric.
# Because attendance is the number of events in which an attendee appears after attendees are admitted
# and before it is too late for new arrivals to be admitted, this might be described as a Poisson distribution.
# Thus a model estimator must be included that does not assume a different distribution.

# Accordingly, the days_before might have an exponential distribution because it describes the amount of time
# that it takes for a fixed (per case) number of poisson events to occur.

# The pair plot gives us some idea of possible relationships between dependent and independent variables.

###############################################################
# Model Fitting
# Choose and fit a baseline model
# Choose and fit a comparison model

# Start coding here...


###############################################################
# Model Evaluation
# Mean squared error is chosen as a metric to measure the aggregate distant
# between predictions and actual observations and to highlight the variables that stand out in terms of such distance.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# We have to scale the columns selected for the linear regression model since it requires features to be similarly distributed.
# We'll use min-max scaling to avoid the assumption that any variable is distributed normally even though that's what linear regression requires.
scaler = MinMaxScaler()
scaler.fit(df[model_columns])
scaled_numpy_ndarray = scaler.transform(df[model_columns])
df_scaled = pd.DataFrame(scaled_numpy_ndarray, columns = model_columns)
df_dumb = pd.get_dummies(df['class_category'], prefix='cat', drop_first=True)
df.drop('class_category', axis=1, inplace=True)

N = df.shape[0]
X = pd.concat([df_scaled[model_columns],df_dumb], axis=1)
X.drop('attendance', axis=1, inplace=True)
y = df.loc[:, 'attendance']
print("y")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, shuffle=True)


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
lm = LinearRegression()
treeModel = DecisionTreeRegressor(random_state=SEED)

models = [('LinearRegression', lm), ('DecisionTreeRegressor', treeModel)]
mse_results = []

for name, model in models:
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  mse_result = mean_squared_error(y_test, y_pred)
  mse_results.append(mse_result)

names = [model[0] for model in models]
print("names:", names)
print("mse_results:", mse_results)
plt.bar(np.arange(len(models)), height=mse_results)
plt.xticks(np.arange(len(models)), names, rotation=20)
plt.ylabel("mean_squared_error")
plt.tight_layout()
plt.savefig('performance.png')
plt.close()


print("X.columns:", X.columns)
# over_6_month has the 2nd greatest slope, but its not normalized as a proportion of attendence
# todo: make a dataframe for these and sort
# - np: argsort, ~maxargs
# todo: include coefficients of DecisionTreeRegressor
# rfr.feature_importances_
# todo: compare alpha and p-value

sns.regplot(data=df, y='attendance', x='age', order=1, x_jitter=0.1, x_estimator=np.mean, ci=True)
plt.savefig('regplot-age.png')
plt.close()

sns.regplot(data=df, y='attendance', x='over_6_month_ratio', order=2, x_jitter=0.1, x_estimator=np.mean, ci=True)
plt.savefig('regplot-seniority.png')
plt.close()

###############################################################
# Communication

# Consistent with the non-normal distribution of some variables as discussed above,
# the linear regression model still obtains a higher mean squared error than the desicion tree model.

# Among the independent variables that are plausiblly normally distributed,
# the student age has the greatest slope, in the linear model.
# This may suggest that company marketing and incentives should target this group, which in turn may influence the total number of attendees.

# Runner ups are over_6_month_ratio, new_students_ratio, and capacity_binary
# todo maybe: to obtain the predictions, we have to undo the transformation to the over_6_month features?
# - Does the best model require scaling and non-multicollinearity? If not, predict again with acceptable raw data.

# For each experienced student that signs up for a class early, an 0.7___ increase is expected for class actual attendance.
# Given the extent of the regplot, it looks like the data may be non-linear, so an order of 2 is specified.

# Since it is plausible that both attendance and number of early sign-ups could logically be dependent variables,
# this case study focuses on actual attendance since its prediction may have the greater value from the standpoint of the company.

# TODO
# Because linear regression is probably not the most appropriate
# and more likely not the most successful model, we can rerun the best model KNN without the scaled features, since this type of regression doesn't need similarly scaled features.
# TODO: Plot train, test and predicted points as a scatterplot in different hues

# TODO: Import from github as a workspace

# TODO: Tune the hyperparameters of the ensenble. These will be unique to each estimator
# - consider: GridSearchCV, ParameterGrid

###############################################################
