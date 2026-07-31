# 5th place solution

My solution focused on time series decomposition and was very similar to [@kdmitrie](https://www.kaggle.com/kdmitrie)'s excellent [approach](https://www.kaggle.com/code/kdmitrie/pgs501-model-1-time-series-decomposition?scriptVersionId=218663573). 

### **1. Decomposition** 
Decompose the time series to remove the effect of:
- Day-of-week 
- Country (GDP)
- Store
- Product
- Day-of-year

### **2. Forecasting**  
After decomposition, I attempted to forecast the remaining curve. This led to a key decision: should the forecast follow the upward trend or remain constant. To test both possibilities, I submitted one version allowing the trend to continue and another assuming a constant trajectory, similar to 2016, with the constant trajectory performing much better. I think this decision probably explains some of the LB shakeup.

### **3. Holidays**  
Incorporating holiday effects had a substantial improvement on leaderboard scores. I used the median of the previous year’s normalised holiday values to estimate the adjustments. I accounted for:  
- Holiday delays 
- Holidays that do not occur on the same day every year
- Three additional holidays in Kenya that were missing from the holiday package

There's probably still some room for improvement here!

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4687599%2F79ffb507d3910b9d0ae0d54209621887%2Fholiday_factor.png?generation=1738371097266680&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4687599%2F4a3c3a42b5b76068f77d44374ded1aec%2Fholiday_factor_2.png?generation=1738371175350648&alt=media)