# 1st Place Solution

Congratulations to @greysky and @hardyxu52 ; you were both formidable competitors. Like Hardy Xu, just when I thought scores may have hit a bottom and there was no more signal left, someone found a way to limbo a little lower. A special thanks to Kaggle and the team at Rohlik for organizing this challenge and answering any dataset queries in the chat. It was truly a great opportunity to apply machine learning to sales data. The dataset was rich in insights and intricacies, and I hope Rohlik are able to find some value in the solutions to help their business. Lastly, I acknowledge the creators of these marvelous machine learning algorithms such as XGBoost and LightGBM. 

**Overall Methodology**

My solution was heavily based upon the M5 Forecasting – Accuracy competition held back in 2020. Many of the winning solutions from that competition used an ensemble of lightGBM models and tended to blend recursive and direct forecasts. Based on this I decided to stick to using GBT algorithms and invested my time heavily in optimizing them. Resources I used to construct my approach:

•	I studied extensively the winning approaches of M5 – in particular the [1st place solution by Yeonjun In ](https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/163684)
•	This write up of the results and findings from[ M5 was quite insightful ](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
•	This blog article was [helpful as well ](https://phdinds-aim.github.io/time_series_handbook/08_WinningestMethods/lightgbm_m5_forecasting.html)

**Model**

I initially started off with a single LightGBM model and decided to use a direct forecast because computationally this would be easier than the recursive method. Early on I decided that rather than have a single direct forecast for all 14 days that I would forecast each day separately to achieve maximum accuracy. I found this approach of 14 forecasts for each day gave a boost of around  ~0.4 over a single 14 day period forecast. 
I found that the results from the single LightGBM model were quite variable, and the model was sensitive to certain features being added/ removed. In the end I found that ensembling with an XGBoost model helped to reduce variability and increase accuracy (boost of  ~0.2). 

**Validation**

To validate my results I used the period from 20th May 24 to 2nd Jun 24 mostly – since this is the two week period immediately prior to the test period. I also used the period from 05th May 24 to 19th May 24 to check but generally I found that if the error improved on the first validation set it would pretty much always improve on the second so to save time, I just validated on the first period only later in the comp. 
I found a decent correlation with Public LB so I picked final submission based on validation score. My final solution scored 17.08 on Private LB and scored 17.81 on Public LB. My best Public LB was 17.59 and would have scored 17.05 on Private but this had a higher validation error so I didn’t pick this for either of my top 2 submissions. 

**Data**

I stuck to the supplied data and based all engineering of this and didn’t use any external data – I just felt it was too risky to use external data given the host comments around its usage and potential for using leaky features. 

**Features**

Feature engineering was critical for this competition and why I enjoyed participating so much. Ones which I didn’t really see used in public NBs which gave a significant boost in score:

•	Mean encoding sales based on orders, weekdays and other attributes
•	Price /Discount of item relative to its category
•	Competing product availability 

**What didn’t work (for me) ?**

I tried many things to try to get #1 on the public LB…But in the end I thought I was beat because I ran out of time and Ideas. I didn’t have much luck with the following:

•	CatBoost – I just couldn’t get the right parameters – maybe because I used  tweedie objective as opposed to RMSE.
•	Outlier removal – removing outliers including outlier time periods such as covid or periods early in a warehouses history just didn’t seem to help 
•	Recursive predictions – I knew this had potential and was part of the M5 winning solution but just couldn’t get this to work. I saw that #2 had success with this approach though 
•	A warehouse /product category specific model to blend with. It always had lower accuracy and didn’t seem to help much when blending in this comp. But this worked well in M5 competition. 

**Conclusion**

In summary based on the other top solutions shared so far + the public NBs - I think my edge was from:

1.	Extensive feature engineering 
2.	14 daily forecast instead of one-shot forecast for all 14 days

The major drawback of my approach is the time required. To generate the FE for all 14 days takes around 1 day (my code is likely not optimal!). Then to create the 28 models for the 14 day forecast is around 1 more day. Changing to a single 14 day forecast could reduce this total time to just 4 hours for all the FE + train/predict but would have resulted in a Private LB score of around 17.4-17.5 which wouldn’t have been good enough to win but would be far more practical. But in a Kaggle comp – time doesn’t matter, just score 😊