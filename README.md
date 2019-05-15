# Project-Conversion
Effective of Facebook Ads in Measuring Sales Conversion

## Data

The data was collected from an anonymous company's social media ad campaign that was made available on Kaggle. The dataset contains 1143 observations and 11 variables. There are a total of 3 unique company campaigns and 691 various facebook campaigns included in this dataset.

The model aims to predict whether the conversion was successful via purchase (1: bought, 0: didn't buy) for an independent facebook user. The independent variables in the model consist of the following:

- ad_id: an unique ID for each ad
- xyz_campaign_id: an ID associated with each ad campaign of the anonymous company
- fb_campaign_id: an ID associated with how Facebook tracks each campaign
- age: age of the person to whom the ad is shown
- gender: gender of the person to whom the ad is shown
- interest: a code specifying the category to which the person’s interest belongs (interests are as mentioned in the person’s Facebook public profile)
- Impressions: the number of times the ad was shown to that person
- Clicks: number of times that person clicked on that ad
- Spent: Amount paid by company to Facebook to show that ad
- Total conversion: Total number of people who enquired about the product after seeing the ad
- Approved conversion: Total number of people who bought the product after seeing the ad

The raw dataset look as follows:

<img width="838" alt="Screen Shot 2019-05-09 at 10 13 43 AM" src="https://user-images.githubusercontent.com/44821660/57460468-26275d00-7243-11e9-9274-b496df972586.png">


### Data Cleaning and Feature Engineering
- Dropped outliers for variables Impressions and Spent. The cut off was at >2,000,000 for Impressions and >500 for Spent (5 rows were dropped as result)
![download (2)](https://user-images.githubusercontent.com/44821660/57461643-67b90780-7245-11e9-9698-01513e0456e6.png)![download (3)](https://user-images.githubusercontent.com/44821660/57461651-6a1b6180-7245-11e9-90e8-49b24d2a2fc6.png)

- Added column 'success' to indicate a binary outcome of whether the ad led to actual sales (1: sale, 0: no sale)
- Replaced xyz_campaign_id column values with string names (campaign 916 = x, 936 = y, 1178 = z)
- Engineered Total_Conversion and Approved_Conversion features. 
    - Total_Conversion was split into 6 bins based on data distribution as "Amount_Clicked"('Amount_Clicked_0','Amount_Clicked_1-5', 'Amount_Clicked_5-10', 'Amount_Clicked_10-20', 'Amount_Clicked_20-50', 'Amount_Clicked_50-100'). 
    - Approved_Conversion was also split into 6 bins as "Amount_Purchased)('Amount_Purchased_0','Amount_Purchased_1-5','Amount_Purchased_5-10', 'Amount_Purchased_10-20', 'Amount_Purchased_20-50', 'Amount_Purchased_50-100')
    
<img width="708" alt="Screen Shot 2019-05-09 at 10 20 44 AM" src="https://user-images.githubusercontent.com/44821660/57460971-22480a80-7244-11e9-924d-579ecb1311ea.png">
- Converted 'interest' to type category
- Made dummy variables out of xyz_campaign_id, gender, interest, age, Amount_Purchased, and Amount_Clicked (resulting in a total of 59 columns, snapshot below)
<img width="958" alt="Screen Shot 2019-05-09 at 10 23 13 AM" src="https://user-images.githubusercontent.com/44821660/57461141-794ddf80-7244-11e9-9bc9-021cbf7dad89.png">


## Exploratory Data Analysis

![download (1)](https://user-images.githubusercontent.com/44821660/57461447-0bee7e80-7245-11e9-9b75-e12dd6c509a1.png)

A pairplot of relationship between the variables as represented through different interests

![download](https://user-images.githubusercontent.com/44821660/57461439-08f38e00-7245-11e9-9233-9c06bf00ce5e.png)

A pairplot of relationship between the variables as represented through different company campaigns

Both pairplots did not divulge as much explanatory information as I had hoped. 

#### Correlation

I proceeded to examine the correlation between variables of greater than 0.75:
<img width="848" alt="Screen Shot 2019-05-09 at 10 31 32 AM" src="https://user-images.githubusercontent.com/44821660/57461783-a484fe80-7245-11e9-99cd-78acaa3fef7c.png">

There is a high correlation between Total_Conversion with Approved_Conversion, as well as Impressions with Clicks and Spent. Examining each independently provided the following:

![download (4)](https://user-images.githubusercontent.com/44821660/57461909-df873200-7245-11e9-839f-8a523a67e6c5.png)
![download (5)](https://user-images.githubusercontent.com/44821660/57461918-e150f580-7245-11e9-85b4-9a130b4efdf2.png)

#### Class Imbalance

Given the categorical nature of my variables, I wanted to explore any class imbalances within my data.

I looked at success and saw a fairly balanced distribution, meaning there are just as many who bought as those who didn't buy:

![download (6)](https://user-images.githubusercontent.com/44821660/57462111-41e03280-7246-11e9-8160-cd4b65dcabb7.png)

Then I looked at the distribution in gender, also fairly balanced:

![download (7)](https://user-images.githubusercontent.com/44821660/57462112-4278c900-7246-11e9-997c-7ad331bf4618.png)

Lastly I looked at distribution in age group, and 30-34 seems to be the prevalent group in the dataset:

![download (8)](https://user-images.githubusercontent.com/44821660/57462113-4278c900-7246-11e9-913b-0f60fc457c9c.png)

#### Who Bought?

Pivoting and multi-indexing my dataframe, I grouped together each categorical group within the features to find out how many of those bought vs. didn't buy:
- gender: F = 269 out of 549 bought, M = 311 out of 589 bought
- age: 30-34 = 223 out of 423 bought, 35-39 = 129 out of 248 bought, 40-44 = 107 out of 210 bought, 45-49 = 121 out of 257 bought
<img width="166" alt="Screen Shot 2019-05-09 at 10 42 45 AM" src="https://user-images.githubusercontent.com/44821660/57462651-40fbd080-7247-11e9-8330-91b413628d2c.png">
<img width="167" alt="Screen Shot 2019-05-09 at 10 42 49 AM" src="https://user-images.githubusercontent.com/44821660/57462657-435e2a80-7247-11e9-9798-6623ab167dbe.png">


![download (9)](https://user-images.githubusercontent.com/44821660/57462666-48bb7500-7247-11e9-9987-2512d2c90e42.png)

![download (10)](https://user-images.githubusercontent.com/44821660/57462668-4bb66580-7247-11e9-8e9a-401242396f04.png)

## Logistic Regression Model

Target variable = 'success'. Test-Train-Split my model with 80% training data and 20% testing data at a random state of 35.

Using Sklearn, I ran a baseline Logistic Regression model to examine the Accuracy and F1 score of my engineered features:

Test Accuracy score:  0.5131578947368421

Test F1 score:  0.6782608695652175

The result was not great. The predictive power of my baseline is ~51%, almost the same as flipping a coin.

### Feature Selection

I proceeded to use PolynomialFeatures in Sklearn to find all possible interaction terms and polynomials terms (to a degree of 3) and ran my Logistics Regression Model with 35,989 features. Scaling the data using StandardScaler, I arrive at the following results:

Test Accuracy score:  0.9780701754385965

Test F1 score:  0.9785407725321887

In the process of running this model, most of my coefficients went to 0 except for 4,385 features.

Creating a Confusion Matrix, I examined only 2 false negatives and 3 false positives:

![download (11)](https://user-images.githubusercontent.com/44821660/57463585-13178b80-7249-11e9-8bff-dcb1191b5fda.png)

And the same was expressed via my ROC curve:

<img width="380" alt="Screen Shot 2019-05-09 at 10 58 02 AM" src="https://user-images.githubusercontent.com/44821660/57463733-5671fa00-7249-11e9-9914-2f9b96ac21ae.png">


## KNN Model

Scaling my original cleaned dataframe with dummy variables, my baseline KNN model provided these results:

Accuracy:0.8491228070175438

F1: 0.8501742160278745

![download (12)](https://user-images.githubusercontent.com/44821660/57464136-3262e880-724a-11e9-8909-6069ecb190d8.png)

To improve my KNN model, I searched for the optimal value for K, which turned out to be 131:

![download (13)](https://user-images.githubusercontent.com/44821660/57464259-63431d80-724a-11e9-98d4-4d0ea9025835.png)

Re-running my KNN model with k=131, I achieve the following results:

Accuracy:0.9543859649122807

F1: 0.9550173010380624

![download (14)](https://user-images.githubusercontent.com/44821660/57464338-88d02700-724a-11e9-8613-82d791a7f037.png)

Although the Accuracy and F1 score for my KNN model greatly improved after adjusting the parameter of k, its predictive power is still not as great as my Logistic Regression model.

## Decision Tree Using Random Forest

Using GridSearch, the optimal model derived the following results:

Test Accuracy score:  0.9899749373433584

Test F1 score:  0.9902439024390244

The parameters fed into GridSearch were:

param_grid = {'n_estimators': [100, 200,300,400], 'max_features': [0.25, 0.33, 0.5], 'max_depth' : [5,6,7,8,9], 'min_samples_leaf': [0.03,0.04,0.05,0.06] }

where n_estimators are the number of trees, max_features are the percentage of random trees in each node, max_depth is the maximum level of splits, and min_sample_leaf is the percentage of data to include minimally.

The best parameters were:

{'max_depth': 6, 'max_features': 0.5, 'min_samples_leaf': 0.03, 'n_estimators': 200}

Result tree looks as follows:


<img width="800" alt="Screen Shot 2019-05-09 at 11 47 53 AM" src="https://user-images.githubusercontent.com/44821660/57467324-4f021f00-7250-11e9-8e7d-f851dbcf550c.png">


So far, Decision Tree using Random Forest has provided the best result with the strongest predictive power on my test-data.

## XGBoost

My baseline XGBoost model gave the following results:

Test Accuracy score: 0.994987

Test F1 score: 0.995098

Using GridSearch on my XGBoost model, the best paramters and best scores were:

{'max_depth': 3, 'min_child_weight': 1}

Accuracy: 0.994987

F1: 0.995098

No improvement from the base are observed here.

Ranking the importance of my features, I obtain the following:

<img width="468" alt="Screen Shot 2019-05-09 at 1 12 31 PM" src="https://user-images.githubusercontent.com/44821660/57472735-3435a780-725c-11e9-94fd-c3ceccc3f33c.png">



## Conclusion

Out of the four classification models I ran, the best result in terms of Accuracy and F1 Scores was provided by my XGBoost model, with ~99.5% Accuracy on my test-data for a F1 Score of 0.995.






