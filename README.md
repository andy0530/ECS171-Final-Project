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

### Pre-processing

For our categorical variables, we will encode the set of following categories to be `0~(n_classes-1)`. Additionally, we have listed their numerical values for each unique values:

* `hotel` - `'Resort Hotel': 0`, `'City Hotel': 1`
* `arrival_date_month` - `'January': 0`, `'February': 1`, ... , `'December': 11`
* `success_room_type` - `'True': 0`, `'False': 1`
* `arrival_date_year` - `2015: 0`, `2016: 1`, `2017: 2`

However, for the other categorical variables—`'meal'`, `'market_segment'`, `'distribution_channel'`, `'deposit_type'`, `'customer_type'`, `'reservation_status'`— we decided to do one-hot encoding as the unique values of their corresponding columns were not instinctively numerically orderable.

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

### Preprocessing

### Model 1: Logistic Model

### Model 2: Neural Net Model

## Conclusion

## Collaboration
* Jun Ha (Andy) Lee
* Erick S. Arenas
* Soumyajit (Sam) Chatterjee
* Joyjit (Joy) Chatterjee
