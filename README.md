# Jumia Smartphone Price Prediction: Optimizing Retail Strategies through Data Analytics

![Jumia Black Friday GIF](https://media.giphy.com/media/DCCDShls7lYyiPLz7l/giphy.gif)

## Table of Contents
***
1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Metrics of Success](#metrics-of-success)
4. [Data Understanding](#data-understanding)
5. [Data Preparation](#data-preparation)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
8. [Hyothesis Testing](#hypothesis-testing)
7. [Data Pre-processing](#data-pre-processing)
8. [Modeling](#modeling)
9. [Deployment](#deployment)
10. [Conclusion](#conclusion)

## Project Scope
***

![architecture](images/Architecture.jpg)

## Project Overview
***

This project aims to develop a predictive pricing model for Jumia retailers to optimize pricing strategies, especially during high-demand events like Black Friday. By leveraging historical sales data, competitor pricing, and market trends, the model automates price adjustments, enabling retailers to stay competitive and boost sales. The goal is to provide data-driven tools that help retailers navigate the dynamic pricing challenges in Africa's growing e-commerce market.

## Objectives
***

**Main Objective:** To Develop a predictive model that accurately forecasts the optimal market price of smartphones based on key product features.

Other objectives are;
>- Conduct an in-depth analysis of the dataset to identify key features that impact smartphone pricing on the Jumia platform.
>- Analyze the smartphone market in detail to determine the most dominant smartphone brand on the Jumia platform.

## Metrics of Success
***

To evaluate model performance, we will focus on the following metrics:

**Mean Absolute Error (MAE):** Measures the average magnitude of prediction errors, providing an intuitive sense of the average price difference between predicted and actual values.

**Mean Squared Error (MSE):** Emphasizes larger errors, which helps to identify models that avoid significant deviations in price prediction.

**R-Squared (R²):** Indicates how well the model explains the variance in the target variable (Price). A higher R² value represents a better fit, with the model capturing more of the price variation.

Success Criteria:
* The model that minimizes MAE and MSE while maximizing R² will be considered optimal. However, a balance between MAE and R² is key, as MSE may disproportionately penalize larger errors, especially in the case of noisy data.
* The best model will not only deliver low error metrics but will also be actionable for retailers on Jumia, enabling them to adjust prices quickly and accurately in response to changing market conditions during critical sales periods.

## Data Understanding
***

The data for this project was scraped from the Jumia Kenya platform on October 31, 2024, focusing on 12,000 smartphones listed by popularity. The scraping process used Beautiful Soup and Pandas and the data was saved in a CSV file `jumia_phones.csv`. For a detailed look at the web scraping process, refer to the [Web Scraping Script](Scrapped_data.ipynb) The data is accessible for review in the Data Repository [Data](https://github.com/N-kioko/Smartphones_Price_Prediction_and_Discount_Analysis_Project/blob/main/Data/jumia_phones.csv)

## Data Preparation
***

This process involves feature extraction from the scrapped data and data cleaning.

#### Feature Extraction

The original dataset has the Phone Name column containing various phone features that are key to the analysis process. We process the product dataset (df) to extract key features from the Name column using regex. The features extracted include: **Brand**, **Screen Size**, **RAM**, **ROM**, **Color**, **Warranty**, **Camera**, **Battery Power**, **Number of SIMs** (based on the presence of "dual").
These extracted features are then combined into a new DataFrame (final_df) alongside the original data.

 #### Data Cleaning 

This step involves dealing with the missing values.
* The most notable missing data includes the **Old Price** and **Discount** columns.
* Moderate Missing Values: **Screen Size**, **Color**, **Camera**, **Battery Power**, and **Rating** have missing values, but they are not as extensive as the above columns.
* No Missing Values: Columns such as **Brand**, **RAM**, **ROM**, **Price**, **Number of Reviews**, **Search Ranking**, **Page**, and **Rank** have no missing values, which are positive for model development as they can be directly used in analysis.

## Exploratory Data Analysis (EDA)
***

The focus of this analysis is on univariate and bivariate relationships with price as the target variable. By examining correlations between price and features such as brand, specifications (e.g., screen size, RAM, battery power), and ratings, we aim to uncover patterns that influence pricing. This helps identify factors driving price differences across brands and offers insights into the smartphone market's pricing dynamics.

![Correlation matrix](https://github.com/N-kioko/Jumia_Smartphones_Price_Prediction-_Optimizing_Retail_Strategies_through_Data_Analytics/blob/main/images/Heatmap.png)

## Hypothesis Testing
***

### Hypothesis 1

Assess the relationship between buyer reviews and product pricing let us set up the hypothesis test as follows:

Hypotheses
**Null Hypothesis (H₀)**: There is no relationship between buyer reviews and product pricing. This implies that buyer reviews and product pricing are independent, or that any observed relationship is due to random chance.

**Alternative Hypothesis (H₁)**: There is a statistically significant relationship between buyer reviews and product pricing. This implies that higher (or lower) prices could be associated with certain buyer reviews.

##### **Findings** 
Statistical Significance: The p-value from the test is 0.0000, which is well below the significance threshold of 0.05. This means we can reject the null hypothesis and conclude that there is a statistically significant relationship between product pricing and the number of buyer reviews.
Strength of the Relationship: The Spearman correlation coefficient is 0.1536, indicating a weak positive relationship. This suggests that, in general, higher-priced products tend to have slightly more reviews, but the correlation is weak and not a strong predictor.

#### **Conlusion:**
This is an indication that price alone has minimal influence on review volume. Instead, factors like product quality, marketing, and brand reputation likely play a larger role. Businesses should prioritize improving product visibility, marketing efforts, and customer experience over relying on pricing strategies to increase reviews. A holistic approach that includes advertising and promotions can be more effective in driving both purchases and reviews.

### Hypothesis 2

To test whether there is a relationship between the product search rank position and the Number of reviews.
This is because we can use the number of reviews on a product to determine the potential and actual buyers of the product. 

Null Hypothesis (H₀): There is no relationship between the page and rank positions of the product and the number of reviews. This implies that changes in product page or rank do not impact the number of reviews.

Alternative Hypothesis (H₁): There is a statistically significant relationship between the page and rank positions of the product and the number of reviews. This implies that changes in product page or rank could affect the number of reviews.

##### **Findings** 
Rank Impact: The negative relationship between Rank and the number of reviews suggests that products with higher ranks (e.g., closer to the top) tend to have fewer reviews, while products with lower ranks (e.g., further down the list) tend to have more reviews. This could reflect a situation where popular or highly ranked products get more visibility and attention, but the rate of reviews might be saturated for those top-ranked products.

Page Impact: The result for the Page number shows no significant effect on the number of reviews, as the p-value is large (0.504). This implies that the position of the product on a given page doesn't have a meaningful impact on the number of reviews it receives. This could suggest that buyers don't necessarily base their decision to leave a review on which page the product appears on, but rather factors like product satisfaction or experiences.

#### **Conlusion:**
Rank is a statistically significant predictor of the number of reviews, meaning that changes in product rank are related to changes in review counts.
Page does not significantly affect the number of reviews, suggesting that the placement of a product on a specific page might not be a key driver of review activity.
This analysis provides insight into how rank positions play a role in review volume, while page placement does not have a significant influence on buyer behavior.

## Data Pre-Processing
***

Before training the model, it's essential to ensure the data is clean, well-structured, and ready for analysis. The pre-processing steps outlined below help prepare the data, ensuring the model can learn effectively and provide accurate predictions.

We followed this steps to avoid Data leakage:

**Split the Data:** Divide the dataset into training, validation, and test sets to properly evaluate the model's performance. The training set is used to train the model, the validation set helps with hyperparameter tuning, and the test set provides an unbiased evaluation of the final model.

**Handle Categorical Data:** Convert categorical data into a format that the model can interpret. This allows the model to capture relationships between categories, which enhances overall model performance.

**Standardize the Data:** Standardizing features to a common scale improves model convergence speed, stability, and performance by preventing features with larger ranges from disproportionately influencing the model.

## Modeling
***

We began with a baseline linear regression model to predict phone prices based on brand, where the independent variables are defined in X and the dependent variable (price) is y.

#### Linear Regression Model

**Baseline MAE (9.29e-12)** suggests overfitting, as the model fits the training data too closely.
**Cross-Validated MAE (0.0587)** provides a more realistic estimate of performance on unseen data.

###### Regularization: Ridge and Lasso

**Ridge regularization (L2)** was applied to reduce overfitting by penalizing large coefficients. This improved generalization, bringing the MAE to 39.05, a more realistic value.

###### Hyperparameter Tuning

Tuning the alpha parameter in Ridge regression improved the model, with an optimal MAE of 0.81.

A residual plot suggested a non-linear relationship, prompting us to explore polynomial regression.

#### Polynomial Regression

The unregularized polynomial model overfitted, while the regularized model (MAE of 0.414) reduced overfitting and improved generalization.

#### Decision Tree Regressor
The decision tree showed high R-squared values (0.998+), suggesting overfitting, as it fitted both training and validation data very closely.

#### Random Forest
The Random Forest model showed high accuracy and minimal error, with hyperparameter tuning yielding the best model. The validation MAE was 0.21, and R² was nearly perfect (0.9999).
Despite excellent performance, residual analysis indicated slight issues with heteroscedasticity and outliers.

#### Gradient boosting

Gradient Boosting achieved near-perfect performance:
**Training MAE:** 23.63
**Validation MAE:** 23.70
**R²:** 0.99996, indicating minimal error and excellent generalization across both training and validation sets.

#### Extreme Gradient Boost

Extreme Gradient Boosting showed almost identical performance to Gradient Boosting, with extremely low errors and near-perfect R² scores on both training and validation data, confirming its ability to generalize well.

## Model of Choice
*** 

Our preferred model, XGBoost, shows excellent performance with very low error rates and high predictive accuracy.

**MAE (Mean Absolute Error)** is close to 0.001 across training, validation, and test sets, indicating minimal prediction error and strong generalization.
**MSE (Mean Squared Error)** values are extremely low, confirming efficient error minimization with minimal difference between training and test sets.
**R-squared (R²)** values are near 1 (0.9999), indicating that the model explains nearly all the variance in the data and performs consistently across datasets.

XGBoost's ability to handle complex data through boosting and regularization, combined with its high accuracy and interpretability, makes it a robust choice for this task. Despite slight performance gains from neural networks in some cases, XGBoost remains highly efficient and effective.

## Findings
***

**Rank Impact:** There is a negative relationship between product rank and the number of reviews. Higher-ranked products tend to have fewer reviews, possibly due to review saturation, while lower-ranked products receive more reviews.

**Page Impact:** The position of a product on a page does not significantly affect the number of reviews it receives (p-value = 0.504), suggesting that factors like product satisfaction, rather than page placement, influence review frequency.

## Conclusion
***

Rank is a significant predictor of review count, with changes in product rank correlating to changes in review volume. However, product placement on a page does not significantly impact the number of reviews, indicating that rank plays a more important role in driving review activity than page position.

## For More Information
***

See the full analysis in the [Juptyer Notebook](index.ipynb) or review the [Presentation](Presentation.pdf)