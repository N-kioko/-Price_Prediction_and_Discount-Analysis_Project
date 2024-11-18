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

****
## Project Overview
***

This project aims to develop a predictive pricing model for Jumia retailers to optimize pricing strategies, especially during high-demand events like Black Friday. By leveraging historical sales data, competitor pricing, and market trends, the model automates price adjustments, enabling retailers to stay competitive and boost sales. The goal is to provide data-driven tools that help retailers navigate the dynamic pricing challenges in Africa's growing e-commerce market.

****
## Objectives
***

**Main Objective:** To Develop a predictive model that accurately forecasts the optimal market price of smartphones based on key product features.

Other objectives are;
>- Conduct an in-depth analysis of the dataset to identify key features that impact smartphone pricing on the Jumia platform.
>- Analyze the smartphone market in detail to determine the most dominant smartphone brand on the Jumia platform.

****
## Metrics of Success
***

To evaluate model performance, we will focus on the following metrics:

**Mean Absolute Error (MAE):** Measures the average magnitude of prediction errors, providing an intuitive sense of the average price difference between predicted and actual values.

**Mean Squared Error (MSE):** Emphasizes larger errors, which helps to identify models that avoid significant deviations in price prediction.

**R-Squared (R²):** Indicates how well the model explains the variance in the target variable (Price). A higher R² value represents a better fit, with the model capturing more of the price variation.

Success Criteria:
* The model that minimizes MAE and MSE while maximizing R² will be considered optimal. However, a balance between MAE and R² is key, as MSE may disproportionately penalize larger errors, especially in the case of noisy data.
* The best model will not only deliver low error metrics but will also be actionable for retailers on Jumia, enabling them to adjust prices quickly and accurately in response to changing market conditions during critical sales periods.

****
## Data Understanding
***

The data for this project was scraped from the Jumia Kenya platform on October 31, 2024, focusing on 12,000 smartphones listed by popularity. The scraping process used Beautiful Soup and Pandas and the data was saved in a CSV file `jumia_phones.csv`. For a detailed look at the web scraping process, refer to the [Web Scraping Script](Scrapped_data.ipynb) The data is accessible for review in the Data Repository [Data](https://github.com/N-kioko/Smartphones_Price_Prediction_and_Discount_Analysis_Project/blob/main/Data/jumia_phones.csv)

****
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

****
## Exploratory Data Analysis (EDA)
***

The focus of this analysis is on univariate and bivariate relationships with price as the target variable. By examining correlations between price and features such as brand, specifications (e.g., screen size, RAM, battery power), 
nd ratings, we aim to uncover patterns that influence pricing. This helps identify factors driving price differences across brands and offers insights into the smartphone market's pricing dynamics.

![Correlation matrix](https://github.com/N-kioko/Jumia_Smartphones_Price_Prediction-_Optimizing_Retail_Strategies_through_Data_Analytics/blob/main/images/Heatmap.png)

****
## Hypothesis Testing
***
Two hypotheses were tested to examine relationships between buyer reviews, product pricing, and search rank. For pricing,
the null hypothesis was rejected, as a p-value of 0.0000 indicated a statistically significant but weak positive relationship
(Spearman coefficient = 0.1536), suggesting price has minimal influence on review volume compared to factors like product 
quality and marketing. For search rank, a significant negative relationship was found between rank and reviews, with 
lower-ranked products receiving more reviews, possibly due to saturation of reviews for top-ranked items. However,page 
position showed no significant impact on review volume (p-value = 0.504). Overall, rank affects review activity, 
while pricing and page placement have weaker roles, emphasizing the importance of visibility, marketing, and customer
experience in driving reviews.

****
## Data Pre-Processing
***

Before training the model, it's essential to ensure the data is clean, well-structured, and ready for analysis. The pre-processing steps outlined below help prepare the data, ensuring the model can learn effectively and provide accurate predictions.

We followed this steps to avoid Data leakage:

**Split the Data:** Divide the dataset into training, validation, and test sets to properly evaluate the model's performance. The training set is used to train the model, the validation set helps with hyperparameter tuning, and the test set provides an unbiased evaluation of the final model.

**Handle Categorical Data:** Convert categorical data into a format that the model can interpret. This allows the model to capture relationships between categories, which enhances overall model performance.

**Standardize the Data:** Standardizing features to a common scale improves model convergence speed, stability, and performance by preventing features with larger ranges from disproportionately influencing the model.

****
## Modeling
***

We began with a baseline linear regression model to predict phone prices based on brand, where the independent variables are defined in X and the dependent variable (price) is y.

### *Linear Regression Model*
***

**Baseline MAE (9.29e-12)** suggests overfitting, as the model fits the training data too closely.
**Cross-Validated MAE (0.0587)** provides a more realistic estimate of performance on unseen data.

### K-Folds On Baseline Model
Cross validation provides a comprehensive understanding of our model’s performance because it tests the model on multiple sets of data reducing the risk of our model’s performance being overly optimistic or pessimistic based on one split.
In summary it helps detect overfitting or underfitting by observing how the model performs across multiple different subsets.

### *Polynomial Regression*
***
The unregularized polynomial regression model overfitted the data, achieving low errors on training (MAE: 2.95, R²: 0.99997)
and validation (MAE: 2.69, R²: 0.99999) but raising concerns about generalization. Applying Ridge regularization mproved 
generalization by balancing the fit, increasing training MAE slightly (from 2.95 to 3.08) and validation MAE (from 2.69 to 
3.20). These results indicate reduced overfitting, as the regularized model performs better on unseen data. 
The Ridge-regularized model, with cross-validated MAE values (Training: 3.08, Validation: 3.20), is a more robust choice,
demonstrating improved predictive accuracy and generalization.

### *Decision Tree Regressor*
***
The Decision Tree Regressor effectively captures non-linear relationships, outperforming the regularized Polynomial 
Regression model on validation data with a lower MAE (2.65 vs. 3.20). It demonstrates strong performance, with high R² 
scores for both training (0.99997) and validation (0.99999), making it the preferred model due to better generalization. 
While Ridge Polynomial Regression mitigates overfitting, it may not capture complex patterns as well as the Decision Tree. 
To further reduce overfitting and enhance performance, the next step is to implement a Random Forest, which leverages 
ensemble learning to improve accuracy and robustness.

### *Random Forest*
***
he Random Forest Regressor demonstrates performance nearly identical to the Decision Tree model, with similar MAE (2.65), 
MSE, and R² values on both training and validation sets, indicating excellent predictive accuracy and generalization. While
both models perform exceptionally well, the Random Forest is preferred for its added stability and robustness, making it
better suited for handling more complex data in future applications.

#### Hyperparameter Tuning The Random Forest
***
Hyperparameter tuning for the Random Forest Regressor yielded the best parameters (max_depth=None, max_features='sqrt', 
min_samples_leaf=1, min_samples_split=2, n_estimators=300). The tuned model's performance closely mirrors the baseline, 
with minimal differences in training and validation MAE (~2.91 and ~2.66), MSE, and R² (~0.999). The next step is testing 
the model on unseen data to evaluate its real-world performance.

#### Results on Test Data

**Results**
Test MAE: 3.234590638120747  
Test MSE: 154.96864882228948  
Test R-squared: 0.9999928582376032  

The MAE on the test data is 3.23, which is close to the validation MAE of 2.66 and training MAE of 2.91. 
This indicates that the model's predictions are relatively accurate and consistent across all datasets (training, validation,
and test).MSE (Mean Squared Error): The MSE of 154.97 on the test data is also consistent with the validation(126.21) and training MSE (557.66). It’s possible that the MSE for the training set is artificially high due to model 
overfitting or because it might not be the optimal degree for the polynomial. The validation and test MSE values being closer to each other is a good sign that the model generalizes well to unseen data, but the discrepancy with the training MSE suggests that the model could be overfitting the training data.

### *Gradient boosting*
***  

Gradient Boosting and Random Forest models show very similar performance on test data. Gradient Boosting achieved a test MAE
of 3.23, MSE of 154.93, and R² of 0.9999, while Random Forest showed slightly better results with marginally lower MSE and
faster training time. Both models capture nearly 99.89% of the variance in the test data, indicating high prediction 
accuracy. Despite the minimal differences, Random Forest is preferred due to its slightly better performance and efficiency.

### *Extreme Gradient Boost*

***
Extreme Gradient Boosting (XGBoost) and Random Forest models exhibit nearly identical performance on both training and test 
data. XGBoost has a slight edge in test MSE (23,776.80 vs. 23,778.22 for Random Forest), but the difference is negligible. 
Similarly, the test MAE for XGBoost is marginally higher (53.90 vs. 53.89 for Random Forest). Both models achieve comparable
R² scores, indicating excellent predictive accuracy. Given the minimal differences, either model could be used, with 
preference based on specific implementation needs or computational efficiency

### *Neural Networks*
***

**Results**  
Neural Network Performance:  
Train - MAE: 78.98704747178819, MSE: 12104.451030152719, R²: 0.9994578761211759  
Val - MAE: 78.3747886827257, MSE: 11375.729186450508, R²: 0.9995106754877002  
**Summary**  
The results of the neural networks is worse than even our baseline linear regression model. Possible reasons could be due to the fact that Neural networks tend to require large amounts of data to perform well.  With a dataset size of only around 12,000, a neural network may not perform as well as simpler models like random forests or even linear models, which can generalize better on smaller datasets due to their lower complexity.

****
## Model of Choice
****

MAE: Random Forest (51.85) outperforms XGBoost and Gradient Boosting (both 51.88), with a marginal difference.
MSE: Random Forest has the lowest MSE (24,035.17), slightly better than Gradient Boosting and XGBoost, which are around 24,065.98.
R-squared: All three models achieved an almost identical R-squared of 0.99889, explaining 99.89% of the variance in the test data.

##### Conclusion

All three models are comparable in performance, but Random Forest edges out slightly due to lower MAE and MSE. 
For computational efficiency, Random Forest is preferable, as it typically trains faster than the boosting
models. However, XGBoost and Gradient Boosting remain good options if stability and ensemble robustness are prioritized.

***
### Conclusions & Recommendations
***
* There is a statistical relationship between the product rank position and the number of reviews.

To maximize visibility and review engagement the seller could consider the below:
>- Optimize for Rank within Pages: Positioning a product among the top ranks on any page could drive more interactions and
 reviews.
>- Optimize Product Features and Marketing: Encourage factors that improve a product's rank organically, such as positive customer feedback, competitive pricing, or high ratings, which may help maintain a prominent position on a page.

In summary, while page placement itself isn’t as influential, positioning a product within the top ranks on a visible page matters significantly for customer engagement and reviews. Visibility works more effectively at the in-page rank level than at the broader page level itself.

*  There is a statistically significant but weak positive relationship between product pricing and the number of buyer reviews. 

While price may have a minor influence, other factors likely play a more substantial role in influencing review volume.
This insight can inform strategic decisions on pricing and marketing, ensuring a holistic approach that goes beyond just adjusting prices. Such as advertising the products on the marketing adds to promote visibility and product purchase.

*  Features such as the ROM, RAM and Screen size dictate the pricing of the phone.

****
### Room for improvement
****

The dataset does have limitations offering some key areas for future improvement:

* **Dynamic Pricing:** The price distributions shown are static snapshots, which may not represent current market conditions. Prices on e-commerce platforms fluctuate frequently, so the observed data may only reflect the time of scraping. Using real-time data or setting up periodic data collection would yield more accurate and relevant insights.

* **Incomplete or Inconsistent Data:** Due to the variety of phone models and brands, some listings may lack uniform information (e.g. missing battery details or memory specifications) which could lead to variability in the parsed features. This lack of uniformity might skew comparisons, especially in evaluating value for money. Standardizing data collection or implementing stricter data validation could help address this issue.

* **Unverified Ratings and Reviews:** Any insights or model predictions derived from ratings and reviews may be influenced by unverified or biased feedback. Relying solely on these metrics can misrepresent consumer preferences. Using verified reviews and adding other objective metrics (e.g., sales data) might provide a more balanced assessment.

* **Potential Duplicate Listings:** Duplicate or near-duplicate entries, where multiple sellers list the same model, could distort the perceived popularity and average pricing of certain models. Identifying and consolidating duplicates would improve the accuracy of pricing and ranking statistics.

Addressing these limitations could enhance the reliability of insights derived from the data, leading to a more accurate understanding of price trends, brand popularity, and consumer preferences in the smartphone market.



## For More Information
***

See the full analysis in the [Juptyer Notebook](index.ipynb) or review the [Presentation](Presentation.pdf)