# ECS171 Final Project

Group 26 (Jun Ha (Andy) Lee, Erick S. Arenas, Soumyajit (Sam) Chatterjee, Joyjit Chatterjee)


## Abstract

Our project uses the Hotel Booking Dataset to create a machine learning algorithm to predict the success rate of the reservation. The dataset from Kaggle is originally from the article Hotel Booking Demand Datasets, written by Nuno Antonio, Ana Almeida, and Luis Nunes for Data in Brief, Volume 22, February 2019. This labeled dataset contains 119390 observations for a City Hotel and Resort Hotel from the 1st of July 2015 to the 31st of August 2017. We will be creating an artificial neural network using the independent variables: lead_time, arrival_date_year, arrival_date_month, arrival_date_week_number,  arrival_date_day_of_month, stays_in_weekend_nights, stay_in_week_nights, and adults to predict the dependent variable: is_canceled. Additionally, we will be using Logistic Regression as our second machine-learning method to predict is_canceled using the same aforementioned independent variable. By using Logistic regression and ANN, we hope to find which factors are the most important to predict if a person would cancel or not a reservation in a hotel which is represented in the dataset by the value = 0 when not canceled and value = 1 when canceled.


### Dataset

*[link](https://www.kaggle.com/datasets/mojtaba142/hotel-booking)*

Column Descriptions

* `hotel` - booking information of either 'city hotel' or 'resort hotel'
* `is_canceled` - value indiciating if the booking was canceled (1) or not (0)
* `lead_time` - number of days between the date of booking and the date of arrival
* `arrival_date_year` - year of the arrival date
* `arrival_date_month` - month of the arrival date in 12 `str` categories
* `arrival_date_week_number` - week number of the arrival date
* `arrival_date_day_of_month` - day of month of the arrival date
* `stays_in_weekend_nights` - number of nights guests stayed or booked on the weekends (Saturday or Sunday)
* `stays_in_week_nights` - number of nights guests stayed or booked on the weekdays (Monday to Friday)
* `adults` - number of adults
* `children` - number of children
* `babies` - number of babies
* `meal` - categorical value indicating the meal plan ( `'BB'` - Bed & Breakfast)
* `country` - country of origin
* `market_segment` - market segment designation ( `'TA'` - Travel Agents, `'TO'` - Tour Operators)
* `distribution_channel` - booking distribution channel ( `'TA'` - Travel Agents, `'TO'` - Tour Operators)
* `is_repeated_guest` - value indicating if the booking was from repeated guest (1) or not (0)
* `previous_cancellations` - number of previous booking canceled by customer
* `previous_bookings_not_canceled` - number of previous booking not canceled by customer
* `reserved_room_type` - code of type of room reserved (anonymity reasons)
* `assigned_room_type` - code of type of room assigned (anonymity reasons)
* `booking_changes` - number of changes done to the booking by customer
* `deposit_type` - type of deposit made by the customer ( `'No Deposit'` - no deposit made, `'Non Refund'` - payment made in full, `'Refundable'` - deposit made)
* `agent` - ID of the travel agency (anonymity reasons)
* `company` - ID of the booking company (anonymity reasons)
* `days_in_waiting_list` - number of days the booking was in the waiting list
* `customer_type` - type of customer ( `'Transient'` - booking made not part of group, `'Transient-Party'` - booking is transient, but associated to at least one other)
* `adr` - Average Daily Rate (Sum of all lodging transaction divided by the length of stay)
* `required_car_parking_spaces` - number of parking spaces required by customer
* `total_special_requests` - number of special requests made by customer
* `reservation_status` - status of reservation ( `'Check-out'` - customer has stayed and checked out, `'No-Show'` - customer did not check in, `'Canceled'` - customer cancelled)
* `reservation_status_date` - date of when reservation status was set
* `name` - name of guests (not real)
* `email` - email of guests (not real)
* `phone-number` - phone number of guests (not real)
* `credit_card` - credit card number of guests (not real)



## Data Exploration

In our Data Exploration process, we dropping the following columns due to their consequent reasons:

* `name` - Artifically created by owner of dataset
* `email` - Artifically created by owner of dataset
* `phone-number` - Artifically created by owner of dataset
* `credit_card` - Artifically created by owner of dataset
* `country` - Dataset is insufficient to make assumptions on country-wide scale
* `agent` - Not possible to deciphyer the ID of the travel agency
* `reserved_room_type` - Not possible to deciphyer the ID of the room (replaced by 'success_room_type')
* `assigned_room_type` - Not possible to deciphyer the ID of the room (replaced by 'success_room_type')

As stated above, we created a new column `'success_room_type'` which returns a `True `value when the customer was assigned to the room which they reserved `('reserved_room_type' == 'assigned_room_type')` or `False `when the two values does not match.

Additionally, we analyzed the distribution of all variables using a pairplot for all numerical columns and a pie char for all categorical columns. From the numerical columns, we were able to find that a lot of the columns were right-skewed as more observations had a value of 0. Additionally, our QQ-plot also agrees that the columns have a right-skewed distributions as the plots have a long tail to the right. 

As it is not normally distributed, we will preprocess the data by normalizing.

Lastly, we observed NaN values in 'children' column. However, we have replaced all NaN values with 0 as the majority of values in the 'children' column is 0.


## Pre-processing

As mentioned above, we will be normalizing the data through MinMax normalization.
