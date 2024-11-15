# Jumia Smartphone Price Prediction: Optimizing Retail Strategies through Data Analytics

![Jumia Black Friday GIF](https://media.giphy.com/media/DCCDShls7lYyiPLz7l/giphy.gif)

## Table of Contents
1. [Project Overview](https://github.com/N-kioko/Jumia_Smartphones_Price_Prediction-_Optimizing_Retail_Strategies_through_Data_Analytics/blob/main/index.ipynb#project-overview)
2. [Objectives](https://github.com/N-kioko/Jumia_Smartphones_Price_Prediction-_Optimizing_Retail_Strategies_through_Data_Analytics/blob/main/index.ipynb#objectives)
3. [Metrics of Success](https://github.com/N-kioko/Jumia_Smartphones_Price_Prediction-_Optimizing_Retail_Strategies_through_Data_Analytics/blob/main/index.ipynb#metrics-of-success)
4. [Data Understanding](https://github.com/N-kioko/Jumia_Smartphones_Price_Prediction-_Optimizing_Retail_Strategies_through_Data_Analytics/blob/main/index.ipynb#data-understanding)
5. [Data Preparation](https://github.com/N-kioko/Jumia_Smartphones_Price_Prediction-_Optimizing_Retail_Strategies_through_Data_Analytics/blob/main/index.ipynb#data-preparation)
6. [Data Pre-processing](https://github.com/N-kioko/Jumia_Smartphones_Price_Prediction-_Optimizing_Retail_Strategies_through_Data_Analytics/blob/main/index.ipynb#data-pre-processing)
7. [Modeling](https://github.com/N-kioko/Jumia_Smartphones_Price_Prediction-_Optimizing_Retail_Strategies_through_Data_Analytics/blob/main/index.ipynb#modeling)

## Project Overview

This project aims to create a predictive pricing model for Jumia retailers to optimize pricing strategies, particularly during high-demand periods like Black Friday. By using historical sales data, competitor pricing, and market trends, the model will automate price adjustments, helping retailers remain competitive and improve sales performance. The goal is to provide data-driven tools that empower retailers to navigate dynamic pricing challenges in Africa's rapidly growing e-commerce market.

## Objectives

>- Build a predictive model that uses historical sales data, competitor pricing and market trends to forecast optimal   pricing for smartphone products during high-traffic sales events such as Black Friday.
>- Analyze historical sales and pricing data to identify key factors that drive consumer purchasing decisions and pricing trends.

## Metrics of Success
Developing an optimal model for predicting price (a continuous variable) is a regression problem, so the best model will depend on several factors, including the structure of our data, the relationships between features, and the complexity we are willing to handle.Given the columns available in our dataset (e.g. screen size, RAM, storage, camera, battery power, etc.), different regression models will perform better or worse depending on the nature of the data. Therefore, we will be focusing on the MAE, MSE and R-Squared results as our metrics of gauging success.
A model with lower MSE and MAE while registering R-Squared closer to 1 will be the optimal model.

## Data Understanding

The data for this project was scraped from the Jumia Kenya platform on October 31, 2024, focusing on 12,000 smartphones listed by popularity. The scraping process used Beautiful Soup and Pandas and the data was saved in a CSV file `jumia_phones.csv`. For a detailed look at the web scraping process, refer to the [Web Scraping Script](Scrapped_data.ipynb) The data is accessible for review in the Data Repository [Data](https://github.com/N-kioko/Smartphones_Price_Prediction_and_Discount_Analysis_Project/blob/main/Data/jumia_phones.csv)

## Project Scope

![architecture](images/Architecture.jpg)
## Data Preparation
This processinvolves feature extraction from thescrapped data, data cleaning

#### Feature Extraction
The original Data set has the Phone Name column containing various phone features that are key the analysis process.
In a summary, the above code processes a product dataset (df) to extract key features from the Name column using regex. The features extracted include:
**Brand**, **Screen Size**, **RAM**, **ROM**, **Color**, **Warranty**, **Camera**, **Battery Power**, **Number of SIMs** (based on the presence of "dual").
These extracted features are then combined into a new DataFrame (final_df) alongside the original data.

 #### Data Cleaning 

 ***

 This steps involves dealing with the missing values.
* The most notable missing data includes the **Old Price** and **Discount** columns.
* Moderate Missing Values: **Screen Size**, **Color**, **Camera**, **Battery Power**, and **Rating** have missing values, but they are not as extensive as the above columns.
* No Missing Values: Columns such as **Brand**, **RAM**, **ROM**, **Price**, **Number of Reviews**, **Search Ranking**, **Page**, and **Rank** have no missing values, which are positive for model development as they can be directly used in analysis.
For rows where the 'Old Price' is missing, it is replaced with the current 'Price'. This assumes that if no previous price is recorded, the current price can serve as a reasonable proxy. This ensures that the 'Old Price' field is populated, allowing for accurate Discount calculations.

Missing values in the 'Rating' column are replaced with 0, indicating that the product may not yet have any ratings or reviews. This approach prevents missing ratings from impacting the analysis or model, while 0 can act as a placeholder until actual ratings are available.
The Screen Size data is heavily negatively skewed (-3.52), with most values clustering around larger screen sizes and a long tail toward smaller sizes. Due to this skewness, imputing screen size missing values with the median is more appropriate than the mode, as the median better represents the central tendency and avoids the distortion caused by outliers. Using the mode could lead to a misleading imputation, especially in the case of skewed distributions where outliers have a disproportionate effect.

## Exploratory data 

***

The major focus here is both univariate and bivariate price analysis
In our context, price is the target variable, and we aim to understand its relationship with other features in the dataset. By exploring correlations between price and attributes such as brand, specifications (like Screen Size, RAM, Battery Power), and ratings, we can uncover patterns that influence pricing. This analysis may reveal insights into factors that drive price differences across brands, helping to identify the most value-driven options and providing a clearer understanding of the pricing dynamics in the smartphone market.

## Data Pre-processing
***

Before training the model, it is crucial to ensure that the data is clean, well-structured, and ready for analysis. The pre-processing steps below will help in preparing the data for modeling, ensuring that the model can learn effectively and provide accurate predictions.

We followed this steps to avoid Data leakage:

**Split the Data:** Divide the dataset into training, validation, and test sets to properly evaluate the model's performance. The training set is used to train the model, the validation set helps with hyperparameter tuning, and the test set provides an unbiased evaluation of the final model.

**Confirm Missing Values:** Identify and address any missing values to ensure data consistency which helps improve the model's reliability and performance.

**Handle Categorical Data:** Convert categorical data into a format that the model can interpret. This allows the model to capture relationships between categories, which enhances overall model performance.

**Standardize the Data:** Standardizing features to a common scale improves model convergence speed, stability, and performance by preventing features with larger ranges from disproportionately influencing the model.

## Modeling

***

We shall commence with our baseline model as a linear regression where our variables are as defined in the X (independent variable) and price of the phone as y variable. Our model aims to predict the price of the phone based on the brand. 

#### Linear Regression Model

###### K-Folds on the Baseline Model
Baseline MAE (9.29e-12): Indicates a nearly perfect fit to the training data, likely resulting from overfitting.
Cross-Validated MAE (0.0587): Reflects the model's error on validation folds, providing a more realistic estimate of performance on new data and revealing that the model’s performance is not as perfect outside of the training data.
This comparison highlights that the baseline model is likely overfitted, while the cross-validated MAE is a more reliable measure of generalization error.

###### Applying Ridge and Lasso
Applying Ridge Regularization (also known as L2 regularization) to your baseline model can help control overfitting by adding a penalty to large coefficients, which effectively shrinks them. This approach reduces the model’s tendency to overly fit to the noise in the training data, improving its generalization ability to unseen data.

###### Hyperparameter Tuning
The baseline model has a very low MAE which suggests that the model’s predictions are extremely close to the actual values. An MAE this low is unusual for real-world data, which might indicate overfitting. Thus why we introduced the regularization technique Ridge and trained out model with this. The MAE result is 39.05 which is more realistic in real world. Regularization typically increases error on the training set to help improve performance on unseen data. We shall proceed to do the Hyperparameter tuning on the alpha level to determine the best search. With this in place we were able to tell the best alpha score which gave us an MAE of 0.81kes. 

This residual plot above suggests there may be a non-linear relationship in our data. Ideally in a linear model the residuals are expected to be randomly scattered around zero without any clear pattern. However, in this plot, there is a slight clustering and potential structure among the points, indicating that a linear model may not fully capture the relationship between the features. We shall therefore also venture into polynomial model.

#### Polynomial Regression
Polynomial regression would help capture these kinds of non-linear patterns, which could lead to better accuracy than a linear model. We shall therefore try this as well.
The unregularized polynomial model is overfitting as evidenced by the extremely low MAE and perfect R-squared suggesting it is unlikely to generalize well to new data.
The regularized model, with a cross-validated MAE of 0.414, offers a more reliable performance estimate. Regularization effectively controls overfitting leading to a model that is better suited for generalization.
Regularization improved the model’s robustness by reducing overfitting which should make it more dependable for predictions on unseen data.

#### Decision Tree Regressor
The decision tree's high R-squared on both training and validation sets (0.998+) suggests a strong fit but possibly with some overfitting.

#### Random Forest
The model shows high accuracy with minimal error and excellent generalization across both the training and validation sets. However, the residual plot you shared earlier shows some slight issues, like heteroscedasticity and possible outliers, which the values and errors don’t fully reveal. This could suggest that while the model performs well overall, there might still be specific data characteristics that a Random Forest model doesn't capture perfectly, particularly with heteroscedasticity and outliers.
 ###### hyperparameter tuning the random forest
 Fitting 5 folds for each of 324 candidates, totalling 1620 fits
Best parameters found:  {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
Validation MAE: 0.20897222222221343
Validation MSE: 3.500927339506127
Validation R-squared: 0.9999998494083733
With the above results,The tuned Random Forest model has achieved near-perfect performance with the chosen hyperparameters, showing minimal prediction error and explaining almost all of the variance in the validation data. These metrics suggest that the model is highly accurate, though it may still be worth checking for any overfitting on real-world data despite the impressive

#### Gradient boosting
The Gradient Boosting Regressor model has achieved excellent performance on both the training and validation data. 

Training Performance:
Mean Absolute Error (MAE): 23.63 – On average, the model's predictions on training data deviate from actual values by only 23.63 units.
Mean Squared Error (MSE): 875.33 – This low MSE indicates minimal large errors in training predictions.
R-squared (R²): 0.99996 – The model captures 99.996% of the variance in the training data, showing an extremely high fit.

Validation Performance:
Mean Absolute Error (MAE): 23.70 – On validation data, predictions deviate from actual values by an average of 23.70 units, nearly the same as in training.
Mean Squared Error (MSE): 838.06 – This low MSE on validation data indicates consistency in prediction accuracy.
R-squared (R²): 0.99996 – The model explains 99.996% of the variance on validation data, highlighting its strong ability to generalize.

High Prediction Accuracy: The near-perfect R² score indicates that the model can predict outcomes with exceptional accuracy, aligning closely with actual values.
Reliable Generalization: Consistent metrics between training and validation data demonstrate the model’s ability to maintain accuracy on unseen data, reducing the likelihood of overfitting.
Minimal Prediction Error: With low MAE and MSE values, the model's predictions are precise, minimizing the impact of large errors.

#### Extreme Gradient Boost

High Precision and Consistency: The model’s extremely low error rates (MAE and MSE) and near-perfect R² scores across training and validation data indicate that it is making predictions with almost exact precision.
Generalization Across Data: The similarity in performance metrics between training and validation sets demonstrates the model’s ability to generalize well, suggesting it will perform reliably on unseen data.
Near-Perfect Model Fit: With an R² value nearing 1.0, the model is capable of capturing nearly all the variance in the data, which is exceptionally rare in real-world applications.

## Model of Choice
*** 
Our preferred model is Extreme Gradient Boosting (XGBoost).

The MAE represents the average magnitude of errors between predicted and actual values. The MAE is very low (close to 0.001) across training, validation, and test sets. This suggests that on average, the model is making very small errors in its predictions, which is a strong indicator of high predictive accuracy. The slight increase in MAE from training to test is normal and indicates that the model is generalizing well with minimal overfitting.

MSE measures the average of the squared differences between predicted and actual values and it gives more weight to larger errors. The values are extremely low, confirming that the model is minimizing error efficiently. The slightly higher MSE on the test set (1.58e-06) compared to the training set (1.52e-06) is typical, indicating some degree of generalization, but the difference is minimal, meaning the model performs consistently across different datasets.

R-squared indicates how well the model’s predictions match the actual values. An R² value very close to 1 means that the model is explaining nearly all of the variance in the data, which is exceptional. The values of 0.9999 for training, validation, and test sets suggest that the XGBoost model is very good at predicting the target variable and that it generalizes extremely well.

The XGBoost model is showing excellent performance with very low error and high predictive accuracy. Its ability to generalize well to both the validation and test sets combined with its robust handling of complex data through boosting and regularization makes it an ideal candidate for this task. Even though neural networks might show slightly better performance in some cases, XGBoost is still a strong contender due to its efficiency, interpretability, and ability to handle diverse datasets.

## Hypothesis testing
*** 
#### Hypothesis 1

Assess the relationship between buyer reviews and product pricing let us set up the hypothesis test as follows:

Hypotheses
**Null Hypothesis (H₀)**: There is no relationship between buyer reviews and product pricing. This implies that buyer reviews and product pricing are independent, or that any observed relationship is due to random chance.

**Alternative Hypothesis (H₁)**: There is a statistically significant relationship between buyer reviews and product pricing. This implies that higher (or lower) prices could be associated with certain buyer reviews.

**Findings**

Statistical Significance: The p-value from the test is 0.0000, which is well below the significance threshold of 0.05. This means we can reject the null hypothesis and conclude that there is a statistically significant relationship between product pricing and the number of buyer reviews.

Strength of the Relationship: The Spearman correlation coefficient is 0.1536, indicating a weak positive relationship. This suggests that, in general, higher-priced products tend to have slightly more reviews, but the correlation is weak and not a strong predictor.

Implications for Stakeholders:

Price and Reviews: While the relationship between product pricing and reviews is statistically significant, it is weak. This means that while more expensive products may tend to get more reviews, price alone does not significantly influence the number of reviews. Other factors such as product quality, marketing, or brand reputation may be more impactful in determining review volume.

Consideration for Strategy: The weak correlation suggests that price may not be the primary driver for increasing review counts. Businesses might want to focus on improving product visibility, marketing efforts, or customer experience to boost review numbers, rather than relying solely on pricing strategies.

Conclusion: There is a statistically significant but weak positive relationship between product pricing and the number of buyer reviews. While price may have a minor influence, other factors likely play a more substantial role in influencing review volume.

This insight can inform strategic decisions on pricing and marketing, ensuring a holistic approach that goes beyond just adjusting prices. Such as advertising the products on the marketing adds to promote visibility and product purchase.

#### Hypothesis 2

To test whether there is a relationship between the product search rank position and the Number of reviews.
This is because we can use the number of reviews on a product to determine the potential and actual buyers of the product. 

Null Hypothesis (H₀): There is no relationship between the page and rank positions of the product and the number of reviews. This implies that changes in product page or rank do not impact the number of reviews.

Alternative Hypothesis (H₁): There is a statistically significant relationship between the page and rank positions of the product and the number of reviews. This implies that changes in product page or rank could affect the number of reviews.

**Findings**

Rank Impact: The negative relationship between Rank and the number of reviews suggests that products with higher ranks (e.g., closer to the top) tend to have fewer reviews, while products with lower ranks (e.g., further down the list) tend to have more reviews. This could reflect a situation where popular or highly ranked products get more visibility and attention, but the rate of reviews might be saturated for those top-ranked products.

Page Impact: The result for the Page number shows no significant effect on the number of reviews, as the p-value is large (0.504). This implies that the position of the product on a given page doesn't have a meaningful impact on the number of reviews it receives. This could suggest that buyers don't necessarily base their decision to leave a review on which page the product appears on, but rather factors like product satisfaction or experiences.

Conclusion:
Rank is a statistically significant predictor of the number of reviews, meaning that changes in product rank are related to changes in review counts.
Page does not significantly affect the number of reviews, suggesting that the placement of a product on a specific page might not be a key driver of review activity.
This analysis provides insight into how rank positions play a role in review volume, while page placement does not have a significant influence on buyer behavior