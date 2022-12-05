# ECS171 Final Project

Group 26
* Jun Ha (Andy) Lee
* Erick S. Arenas
* Soumyajit (Sam) Chatterjee
* Joyjit (Joy) Chatterjee


## Introduction

As pandemic related restrictions are fading, the number of people travelling are starting to increase. Hotel businesses are going to be taking advantage of this trend by increasing their advertising and marketing budgets in order to compensate for losses incurred during the pandemic. However, from what we've learned, most of the potential revenue will be lost due to cancellation of reservations. In fact, one study puts the average global cancellation rate of hotel reservations at 40%! 

This cancellation rate was surprising to our group. The prospect of creating a model that could accurately predict whether a reservation was going to be cancelled sounded both exciting and extremely important. After all, there were many benefits for a hotel to know ahead of time whether a reservation was going to be cancelled or not. For example, a hotel could reduce the rates or offer additional perks for customers that were more likely to cancel their reservation, enticing them to stay. Additionally, rooms in a hotel could also be prioritized for customers that are less likely to cancel their reservations, as this could be confirmed revenue for the hotel. 

Our group was also interested in what factors had a higher impact on a reservation being cancelled. These factors, if well documented, could help the hotel business better shift their services to include more customers that were less likely to cancel.

## Methods

### Data Exploration
In our Data Exploration process, we dropping the following columns due to their consequent reasons further explained in the discussion section:

* `name` - Artifically created by owner of dataset
* `email` - Artifically created by owner of dataset
* `phone-number` - Artifically created by owner of dataset
* `credit_card` - Artifically created by owner of dataset
* `country` - Dataset is insufficient to make assumptions on country-wide scale
* `agent` - Not possible to deciphyer the ID of the travel agency
* `reserved_room_type` - Not possible to deciphyer the ID of the room (replaced by 'success_room_type')
* `assigned_room_type` - Not possible to deciphyer the ID of the room (replaced by 'success_room_type')

As stated above, we created a new column `'success_room_type'` which returns a `True` value when the customer was assigned to the room which they reserved `('reserved_room_type' == 'assigned_room_type')` or `False` when the two values does not match.

Additionally, we analyzed the distribution of all variables using a pairplot for all numerical columns and a pie char for all categorical columns. From the numerical columns, we were able to find that a lot of the columns were right-skewed as more observations had a value of 0. Additionally, our QQ-plot also agrees that the columns have a right-skewed distributions as the plots have a long tail to the right. 

As it is not normally distributed, we will preprocess the data by normalizing.

Lastly, we observed `NaN` values in `'children'` column. However, we have replaced all `NaN` values with `0` as the majority of values in the `'children'` column is `0`.

### Pre-processing

For our categorical variables, we will encode the set of following categories to be `0~(n_classes-1)`. Additionally, we have listed their numerical values for each unique values:

* `hotel` - `'Resort Hotel': 0`, `'City Hotel': 1`
* `arrival_date_month` - `'January': 0`, `'February': 1`, ... , `'December': 11`
* `success_room_type` - `'True': 0`, `'False': 1`
* `arrival_date_year` - `2015: 0`, `2016: 1`, `2017: 2`

For the following categorical variables, we will do one-hot encoding.
* `'meal'`
* `'market_segment'`
* `'distribution_channel'`
* `'deposit_type'`
* `'customer_type'`
* `'reservation_status'`

As mentioned above, we scaled our data through MinMax normalization.
As mentioned above, we scaled our data through MinMax normalization.

### Model 1: Logistic Model


 For our logistic regression model we first run a general model with all of our feature excepting `'reservation_status_Canceled', 'reservation_status_Check-Out','reservation_status_No-Show'`. Our y-intercept is `is canceled` and our X is the rest of the features.
 In this case the data was split in to both training and testing data with a 80:20 ratio. 


From the this general logistic regresion model we got an accuracy of 81%. we run a p-value test where at significance 0.05 we would find which have actual correlation in predicting the if a booking is cancel. From this test we find that

* `arrival_date_month`
* `arrival_date_week_number`
* `arrival_date_day_of_month`
* `stays_in_weekend_nights`
* `adults`
* `children`
* `adr`
* `meal_SC`
* `market_segment_Online`
* `distribution_channel_Undefined`

fail to reject the null hypothesis and therefore do not have any relation in predicting the if a booking is cancel our not.
Then we look at our error for both test and training data set. With both being similar enough that the model was correctly 
implement as well as having an error of 0.429 and 0.427 with is low enough for it not to be overfitting

Then we created a new model with just the values we consider relevant. We get the same accuracy of 81% as our original model
meaning that our theory was right by eliminating those features.


Finally we conclude that the three most relevant feature are 

* `previous_cancellations`
* `previous_bookings_not_canceled`
* `required_car_parking_spaces`


with required car parking spaces having the largest coeffiecient. previous cancellations is the second largest and
not previous cancellations the third highest. 

Interpretation:
Going back to our model with our best accuracy. We find that for the three most essential features, required car parking spaces have a higher coefficient which shows a great correlation to not cancel the booking. This is because people who reserve more parking spaces are more likely to not cancel the reservation whereas people that do not have parking spaces have the opposite effect. The second highest is previous_cancellations which indicates that if a person has a previous cancellation, they are most likely to cancel again. Therefore it increases the likelihood of a reservation being canceled. Finally, the third highest feature, previous_boo kings_not _canceled shows that if a person has not canceled previously, then they are most likely to not cancel again in their next booking. This means that a person who has a parking spot reserved and has not canceled before is probably not going to cancel their reservation. On the other hand, if a person has canceled before and they do not have reserved parking spaces, they are likely to cancel the booking.

### Model 2: Neural Net Model

## Results

### Model 1: Logistic Model

### Model 2: Neural Net Model

## Discussion

### Data Exploration

While exploring our data, we initially dropped four columns, `'name'`, `'email'`, `'phone-number'`, `'credit_card'`, that was artificially created by the owner of the dataset as this information had no relevancy to our dataset and could be ignored to predict the cacelation status of each observation.
Additionally, we found three additional columns, `'agent'`, `'reserved_room_type'`, `'assigned_room_type'`, with values that were not decipherable as they were replaced by ID, not explained by the dataset due to anonymity reasons.
Therefore, we removed those three columns, but, we were able to utilize some information to create a new column, `'success_room_type'`, which returns a boolean value based on `'assigned_room_type'` and `'reserved_room_type'`.
By doing so, we are able to extract most information from what is given. However, it is important to note that we might classify `'success_room_type'` as `False` when a customer was assigned a room better than the one reserved.
This shows how a situation could be misinterpreted when a customer was simply offered a free 'upgrade'. Yet, we believe that this data might offer significant information that could be crucial in predicting if the customer cancelled or not.
Lastly, we removed `'country'` as the dataset does not contain equal information on all countries in the world. Therefore, we concluded that the dataset is insufficient to make assumptions on a country-wide scale.

Secondly, from our pariplot, we can observe that most of the numerical variables does not contain negative values. This shows that a 'relu' activation function might be useful in creating our nerual net in latter part of this project.
Additionally, we can observe some numerical variables showing a categorical variable-like distribution. For example, `'arrival_date_year'` represents the year of arrival date from a dataset with hotel bookings between the 1st of July 2015 and 31st of August 2017.
Since it only contains three values of [2015, 2016, 2017], we believe it would be better to consider `'arrival_date_year'` as a categorical variables.
Lastly, we can observe presence outliers from the pairplot such as the value with `'children' = 10`. Therefore, we will observe the effects of these points later in the study.

From our Q-Q plots, we can observe a right-skewed distributions as the plots have a long tail to the right. Additionally, we observe no negative values.
Since, we cannot standardize our data when it is not normally distributed, we selected to do MinMax normalization.
From our pie charts, we do see some values as 'Undefined' is categories such as `meal`, `distribution_channel`, and `market_segment`.
However, we will wait until our model fitting to consider these values as the categories mentioned above may not be relevant.
With finding null values, we observe that the four columns that have NaN values in the `'children'` column also has a value 'Undefined' for the `'distribution_channel'` variable.
Therefore, we decided to drop these four rows in our study as the rows had incomplete information to offer significant data to our models.

### Preprocessing

In our data preprocessing, we decided to encode the set of categories to be 0~(n_classes-1) for `'hotel'`, `'arrival_date_month'`, `'success_room_type'`, and `'arrival_date_year'`.
For `'hotel'` and `'success_room_type'`, we decided to encode this way as there was only two unique values in the columns. Therefore, there was no need for additional encoding.
For the other two columns, we decided to encode this way as each columns represents a consecutive numerical value.
For example, `'arrival_date_month'` has 12 unique values corresponding to each month of the year. Therefore, we encoded this variable from 0~11 in the order of the months.
However, for the other categorical variables—`'meal'`, `'market_segment'`, `'distribution_channel'`, `'deposit_type'`, `'customer_type'`, `'reservation_status'`— we decided to do one-hot encoding as the unique values of their corresponding columns were not instinctively numerically orderable.

### Model 1: Logistic Model

### Model 2: Neural Net Model

## Conclusion

## Collaboration
* Jun Ha (Andy) Lee
* Erick S. Arenas
* Soumyajit (Sam) Chatterjee
* Joyjit (Joy) Chatterjee
