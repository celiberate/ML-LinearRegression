import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import quandl
import datetime
style.use('ggplot')

'''
 We will be using quandl service to get
 share price (Net Asset value incase of  Mutual Funds)
of a company (Mutual Fund).

For complete API documentation, refer to https://www.quandl.com/

If you don't have quandl module installed,
you can do it using pip install quandl
'''


# Dates - Last 3 years from present day

start_date = datetime.datetime.now() - datetime.timedelta(days=3 * 365)
end_date = datetime.date.today()


quandl.ApiConfig.api_key = "YOUR_API_KEY_GOES HERE"

# Get Data From Quandl
df = quandl.get('AMFI/135765', start_date=start_date, end_date=end_date, column_index=1)
print(df)
df = df.reset_index()
prices = df['Net Asset Value'].tolist()
dates = df.index.tolist()

# Convert to 1d Vector as sklearn API works with numpy arrays
dates = np.reshape(dates, (len(dates), 1))
prices = np.reshape(prices, (len(prices), 1))

# Define Linear Regressor Object
regressor = LinearRegression()


# regressor.fit(dates, prices)

# #Visualize Results
# plt.scatter(dates, prices, color='yellow', label= 'Actual Price') #plotting the initial datapoints

# plt.plot(dates, regressor.predict(dates), color='red', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
# plt.title('Linear Regression | Time vs. Price')
# plt.legend()
# plt.xlabel('Date Integer')
# plt.show()

# #Predict Price on Given Date
# date = 10
# predicted_price =regressor.predict(date)
# print(predicted_price)
# print(predicted_price[0][0],regressor.coef_[0][0] ,regressor.intercept_[0])


# Splitting the dataset into the Training set and Test set
xtrain, xtest, ytrain, ytest = train_test_split(dates, prices, test_size=0.33, random_state=42)

regressor.fit(xtrain, ytrain)
ytrainpredict = regressor.predict(xtrain)
# Train Set Graph
plt.scatter(xtrain, ytrain, color='yellow', label= 'Actual Price')
# plotting the initial datapoints

plt.plot(xtrain, ytrainpredict, color='blue', linewidth=3,
         label='Predicted Price')

# plotting the line made by linear regression
plt.title('Linear Regression | Time vs. Price (Training Set)')
plt.legend()
plt.xlabel('Date Integer')
plt.show()

print(r2_score(ytrain, ytrainpredict))

ytestpredict = regressor.predict(xtest)
# Test Set Graph
plt.scatter(xtest, ytest, color='yellow', label='Actual Price')
# plotting the initial datapoints
plt.plot(xtest, ytestpredict, color='blue', linewidth=3, label = 'Predicted Price')
# plotting the line made by linear regression
plt.title('Linear Regression | Time vs. Price (Test Set)')
plt.legend()
plt.xlabel('Date Integer')
plt.show()

print(r2_score(ytest, ytestpredict))
