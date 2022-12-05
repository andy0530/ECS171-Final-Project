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

As stated above, we created a new column `'success_room_type'` which returns a `True` value when the customer was assigned to the room which they reserved `('reserved_room_type' == 'assigned_room_type')` or `False` when the two values does not match. The code to create this columns is shown below:

```
data['success_room_type'] = data.apply(lambda row: row.reserved_room_type == row.assigned_room_type, axis=1)
```

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
* `meal`
* `market_segment`
* `distribution_channel`
* `deposit_type`
* `customer_type`
* `reservation_status`

As mentioned above, we scaled our data through MinMax normalization.
As mentioned above, we scaled our data through MinMax normalization.

### Model 1: Logistic Model

 For our logistic regression model we first run a general model with all of our feature excepting `'reservation_status_Canceled', 'reservation_status_Check-Out','reservation_status_No-Show'`. Our y-intercept is `is canceled` and our X is the rest of the features.
 In this case the data was split in to both training and testing data with a 80:20 ratio. Then using all of our variable we run a logistic model on with our train data. The logistic regression model was 1000 iteration and it run a 'newton-CG' algorithm. We as well create a classification report to get the accuracy of the model. 
 
 
```
X = scaled_data.drop(['reservation_status_Canceled', 'reservation_status_Check-Out','reservation_status_No-Show','is_canceled']
                     ,axis= 1)
y = scaled_data['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=17)
model_log = LogisticRegression(penalty='none', max_iter=1000, solver='newton-cg').fit(X_train, y_train)
predict_model = model_log.predict(X_test)
print(classification_report(y_test,predict_model))
predict_model2 = model_log.predict(X_train)
print(classification_report(y_train,predict_model2))
```

 Once we run our model we run our loss function to find our error. We as well run the same model with k-folds of 7 splits to make sure we are correct on our accuracy.
 
```
# for error loss 
predict_model = model_log.predict_proba(X_test)
print(log_loss(y_test,predict_model))
predict_model2 = model_log.predict_proba(X_train)
print(log_loss(y_train,predict_model2))

# for k-fold

cv = KFold(n_splits= 7, random_state=1, shuffle=True)
scores = cross_val_score(model_log, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```
 
After running our model since we find by looking at our scatter plot that some of the variable may not been contributing to the result we decide to run a p-value test to find the if we can drop some of the variables and improve our model. 

```
scores, pvalues = chi2(X, y)
lst = []
for i in range(0,len(model_log.coef_[0])):
    if pvalues[i] > 0.05:
        print(model_log.feature_names_in_[i],"B"+str(i)+ ':',model_log.coef_[0][i],"\nP-value:",pvalues[i])
    elif pvalues[i] < 0.05:
        lst.append(model_log.feature_names_in_[i])
```


At significance 0.05 we find that we fail to reject the null hypothesis for the variables below.

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


Finally we run our model removing a module above. In this case we again split our data again as well as run our logistic regression with 1000 iteration and 'Newtowns-CG' algorithm. We find a similar accuracy and error as the first model. We as well run k-fold for our sencond model to check if we are using our data correctly.


### Model 2: Neural Net Model

In the beggining, all the important libraries are imported (keras and associated layers).
The training and test data are the split into an 80:20 ratio.

The first attempt at generating a neural net model contains only two hidden layers. The activation function for the first hidden layer is relu, or rectified linear unit, and the layer contains 10 nodes. The activation function for the second layer is sigmoid, and the layer contains 1 node. The default batch size and epoch are used, each of which is 32. The model is then fit to the training data (which intializes intial weights) and used to predict on the testing data.

``` 
modelOne = Sequential()
modelOne.add(Dense(units = 10, activation = 'relu', input_dim = 33)) 
modelOne.add(Dense(units = 1, activation = 'sigmoid', input_dim = 10)) 
opt = keras.optimizers.RMSprop(learning_rate=0.0001) 
modelOne.compile(optimizer = opt, loss = 'binary_crossentropy') 
modelOne.fit(X_train.astype('float'), y_train,verbose=0)
```

Once the predictions are recorded, they are thresholded and used to generate a classification report. 

```
yhat_test = modelOne.predict(X_test.astype(float))
yhat_test = [ 1 if y>=0.5 else 0 for y in yhat_test ]
print(classification_report(y_test, yhat_test))
```

Another layer is then added to the model, between the two layers mentioned above. This layer has a relu activation function and consists of 10 nodes. The model then predicts again on the testing data. Another classification report is generated. 

```
modelTwo = Sequential()
modelTwo.add(Dense(units = 10, activation = 'relu', input_dim = 33))
modelTwo.add(Dense(units = 10, activation = 'relu', input_dim = 10))
modelTwo.add(Dense(units = 1, activation = 'sigmoid', input_dim = 10))
opt = keras.optimizers.RMSprop(learning_rate=0.0001)
modelTwo.compile(optimizer = opt, loss = 'binary_crossentropy')
modelTwo.fit(X_train.astype('float'), y_train,verbose=0)
```

The number of nodes in each of the hidden layers are then increased. Following this is another classification report generation. The code is the same as above.

```
modelThree = Sequential()
modelThree.add(Dense(units = 100, activation = 'relu', input_dim = 33))
modelThree.add(Dense(units = 100, activation = 'relu', input_dim = 100))
modelThree.add(Dense(units = 1, activation = 'sigmoid', input_dim = 100))
opt = keras.optimizers.RMSprop(learning_rate=0.0001)
modelThree.compile(optimizer = opt, loss = 'binary_crossentropy')
modelThree.fit(X_train.astype('float'), y_train,verbose=0)
```

Following, this, the number of epochs are increased are decreased to 10 (from the default 32). Another classification report is generated.

```
modelFour = Sequential()
modelFour.add(Dense(units = 100, activation = 'relu', input_dim = 33))
modelFour.add(Dense(units = 100, activation = 'relu', input_dim = 100))
modelFour.add(Dense(units = 1, activation = 'sigmoid', input_dim = 100))
opt = keras.optimizers.RMSprop(learning_rate=0.0001)
modelFour.compile(optimizer = opt, loss = 'binary_crossentropy')
modelFour.fit(X_train.astype('float'), y_train,epochs=10,verbose=0)
```

Finally, the batch size and number of epochs are changed to reach the final neural net model. Additionally, the number of nodes for the second layer were decreased down to 10. The epochs are changed to 100 and the batch size are now 1000.

```
modelFive = Sequential()
modelFive.add(Dense(units = 1000, activation = 'relu', input_dim = 33))
modelFive.add(Dense(units = 10, activation = 'relu', input_dim = 1000))
modelFive.add(Dense(units = 1, activation = 'sigmoid', input_dim = 10))
opt = keras.optimizers.RMSprop(learning_rate=0.0001)
modelFive.compile(optimizer = opt, loss = 'binary_crossentropy')
modelFive.fit(X_train.astype('float'), y_train, batch_size = 1000, epochs = 100)
```

## Results

### Model 1: Logistic Model


From the this general logistic regresion model we got an accuracy of 81%.Then we look at our error for both test and training data set. Both error rates being similar enough that the model was correctly implement as well as having an error of 0.419 and 0.417. 

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model1_1.png?raw=true)

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model1_2.png?raw=true)

We aswell test our model with k-fold getting an accuracy of 81.4% and std 0.003.

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model1_8.png?raw=true)




we run a p-value test where at significance 0.05 we would find which have actual correlation in predicting the if a booking is cancel. From this test we find that the values below have not significance in our model.

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model1_7.png?raw=true)





Then we created a new model with just the values we consider relevant. We get the same accuracy of 81% as our original model
meaning that our theory was right by eliminating those features. SImilarly for our error we got 0.422 and 0.420 which is close enough to each other to inference that there is not underfitting the model or overfitting. 

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model2_1.png?raw=true)
![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model2_7.png?raw=true)

looking at our k-fold we got an accuracy of 81.3% and std 0.002

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model2_7.png?raw=true)

Finally we conclude that the three most relevant feature are :

* `previous_cancellations`
* `previous_bookings_not_canceled`
* `required_car_parking_spaces`
with coefficient:

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model2_2.png?raw=true)
![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model2_6.png?raw=true)



### Model 2: Neural Net Model

There were a total of 5 attempts made at creating the neural net model. The fifth and final attempt is considered as our final neural net model. Here are the results for each of the attempts, given as classification reports.

Attempt 1 (Simple Neural Net):

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/NN_Attempt1.png?raw=true)

Attempt 2 (Additional Layer):

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/NN_Attempt2.png?raw=true)

Attempt 3 (More Nodes Per Layer):

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/NN_Attempt3.png?raw=true)

Attempt 4 (Decreased Epochs):

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/NN_Attempt4.png?raw=true)

Attempt 5 (Increased Epochs and Batch Size):

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/NN_Attempt5.png?raw=true)

## Discussion

### Data Exploration

While exploring our data, we initially dropped four columns, `'name'`, `'email'`, `'phone-number'`, `'credit_card'`, that was artificially created by the owner of the dataset as this information had no relevancy to our dataset and could be ignored to predict the cacelation status of each observation.
Additionally, we found three additional columns, `'agent'`, `'reserved_room_type'`, `'assigned_room_type'`, with values that were not decipherable as they were replaced by ID, not explained by the dataset due to anonymity reasons.
Therefore, we removed those three columns, but, we were able to utilize some information to create a new column, `'success_room_type'`, which returns a boolean value based on `'assigned_room_type'` and `'reserved_room_type'`.
By doing so, we are able to extract most information from what is given. However, it is important to note that we might classify `'success_room_type'` as `False` when a customer was assigned a room better than the one reserved.
This shows how a situation could be misinterpreted when a customer was simply offered a free 'upgrade'. Yet, we believe that this data might offer significant information that could be crucial in predicting if the customer cancelled or not.
Lastly, we removed `'country'` as the dataset does not contain equal information on all countries in the world. Therefore, we concluded that the dataset is insufficient to make assumptions on a country-wide scale.

![pairplot](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/1.png?raw=true)

Secondly, from our pariplot, we can observe that most of the numerical variables does not contain negative values. This shows that a 'relu' activation function might be useful in creating our nerual net in latter part of this project.
Additionally, we can observe some numerical variables showing a categorical variable-like distribution. For example, `'arrival_date_year'` represents the year of arrival date from a dataset with hotel bookings between the 1st of July 2015 and 31st of August 2017.
Since it only contains three values of [2015, 2016, 2017], we believe it would be better to consider `'arrival_date_year'` as a categorical variables.
Lastly, we can observe presence outliers from the pairplot such as the value with `'children' = 10`. Therefore, we will observe the effects of these points later in the study.


![qq1](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/2-1.png?raw=true)
![qq2](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/2-2.png?raw=true)

(Samples from series of Q-Q plots)

From our Q-Q plots, we can observe a right-skewed distributions as the plots have a long tail to the right. Additionally, we observe no negative values.
Since, we cannot standardize our data when it is not normally distributed, we selected to do MinMax normalization.

![pie1](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/2-3.png?raw=true)
![pie2](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/2-4.png?raw=true)
![pie3](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/2-5.png?raw=true)

(Samples from series of pie charts)

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
When we are looking to predict the probability of semthing happening. An accuracy of 90% is preferable the accuracy we got for 81% is significant enough that the data provide would can actully be used to determine if a cancellation of an hotel booking would be done.
For our error result for our first model we find that with is similar enough error for it not to be overfitting and not underfitting as well.

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model1_6.png?raw=true)

looking at our confussion table we find that we are better at preddicting if a booking was not cancel with f1 score of 86% compared to predicting if they cancel with 72%

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model1_5.png?raw=true)

this meaning that our data is better and predicting when a booking would be cancel compared to when a booking would not be cancel.

The reason for using the p-value test in the model is to find if we can improve our model by removing the least significant part of the data. Meaning that by reducing the variable we are most likely to find a better since we dismish the classification issue that can be cause by some of this variable. After running our second model and finding a similar result with accuracy of 81% we think that the eliminating the variables was good for our model but it did not improve it at all. Which actually show that the data in the variable we drop is actually not useful for the model.

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model2_5.png?raw=true)

Similarly for our second model we find that the error is closed enough for it not to be overfitting our underfitting our model. looking at our confussion table we find that we are better at preddicting if a booking was not cancel with f1 score of 86% compared to predicting if they cancel with 72%

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model2_3.png?raw=true)


Finally we find that the largest coeffiecient in the model to be: required car parking spaces having the largest coeffiecient. previous cancellations is the second largest and not previous cancellations the third highest. 

![png](https://github.com/andy0530/ECS171-Final-Project/blob/main/figures/Picture_model2_7.png?raw=true)

Interpretation:
Going back to our model with our best accuracy. We find that for the three most essential features, required car parking spaces have a higher coefficient which shows a great correlation to not cancel the booking. This is because people who reserve more parking spaces are more likely to not cancel the reservation whereas people that do not have parking spaces have the opposite effect. The second highest is previous_cancellations which indicates that if a person has a previous cancellation, they are most likely to cancel again. Therefore it increases the likelihood of a reservation being canceled. Finally, the third highest feature, previous_boo kings_not _canceled shows that if a person has not canceled previously, then they are most likely to not cancel again in their next booking. This means that a person who has a parking spot reserved and has not canceled before is probably not going to cancel their reservation. On the other hand, if a person has canceled before and they do not have reserved parking spaces, they are likely to cancel the booking.


### Model 2: Neural Net Model

The nerual net was chosen as the second model for its ability to handle complex relations between the features and the label (is_cancelled). 

We began by  adding two hidden layers because we didn't want to increase the complexity of the model. Rather, we wanted to start with a simpler neural nets and adjust the hyperparameters along the way. 

We chose the relu activation function because of the way it handles negative values. For example, it would be odd to think that a parent could have a negative number of children. The relu function realistically handles this case by assigning a bottom boundary of 0.

Our choice of the sigmoid activation function was influenced by our decision for the output to be a probability, between 0 and 1.

The output was thresholded in order to get discrete outputs of 0 or 1. This was required since the classification report generates recall, precision, and accuracy based on two arrays (yhat_test and y_test) that contain discreet values that are either 0 or 1. The threshold was set to .5 because we wanted to round our probabilities (< .5 gets rounded to 0, and > .5 gets rounded to 1).

The accuracy for the first attempt wasn't great and we decided that a more complex model could increase our accuracy. Therefore, the second hidden layer (of relu activation) was added. However, that didn't seem to help the accuracy or precision of our model either. In fact, the accuracy for the first and second attempts were almost the same. We added more nodes to the hidden layers of the model, but it didn't increase the accuracy by too much. Therefore, we decided that increasing the complexity of our model anymore wasn't going to help and decided to tune other hyperparameters.

Attempts 4 and 5 engaged in changing the number of epochs and batch size. Our reasoning was based on the assumption that maybe the model was complex enough, but it's cross validation was poor. Therefore, in the fourth attempt, the number of epochs were decreased to make sure that the model would run less times on the training data. In the fifth attempt, the batch size was increased to make sure that the model calculated loss after seeing many observations at the same time. 

After 5 attempts, the neural net model gave a pretty high accuracy. However, the final accuracy was still not above 90. We think that changing the different activation functions could have made a model with greater predictibility, but were limited by our knowledge of what combinations to use. Additionally, maybe adding more nodes could have increased our accuracy slightly, but it clearly didn't show a huge impact on overall predicitibility of the model. 

## Conclusion

## Collaboration
While there was no team leader/manager, we initially divided the project into four parts to be handled by each member
* Jun Ha (Andy) Lee - worked on data exploration/pre-processing
* Erick S. Arenas - worked on Model 1: Logistic Model
* Soumyajit (Sam) Chatterjee - worked on Model 2: Neural Net Model
* Joyjit (Joy) Chatterjee - worked on report/readme and helped on Model 2: Neural Net Model
