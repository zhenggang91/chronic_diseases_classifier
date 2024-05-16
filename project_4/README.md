<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# Project 4 - Unveiling Chronic Disease in Singaporean Lifestyle

> Authors: Chung Yau, Gilbert, Han Kiong, Zheng Gang
---

## Overview

This project targets the relationship between lifestyle habits and chronic disease risks. Our mission is to leverage data science for identifying individuals at high risk based on their behaviors, aiming for early detection and personalized health strategies.

We'll analyze lifestyle data, such as alcohol consumption, smoking habits, dietary patterns, physical activity, and other health metrics, to pinpoint risk factors. Through data cleaning and preparation, we set the stage for exploratory data analysis (EDA) and predictive modeling. Our objective is to offer tailored recommendations, empowering individuals to make informed decisions. This approach embodies a proactive stance towards health, using data-driven insights to foster a healthier society by mitigating chronic disease risks.

After prediction, we also work on a food recommender that recommend dishes that suits the users' nutrional requirements. The food recommender is in a beta version as there are limited nutritional data available online. **Please note that the focus on the recommender is on the implementation, and all recommendations should be double confirmed by a certified medical professional before going live.**

## Persona

Conrius, a 30-year-old auditor at KPMG, is mindful of the health risks associated with his foodie hobby amidst his strenuous work schedule. He seek personalized guidance, not generic online information, 
Conrius aims to assess his lifestyle's impact on potential chronic diseases. 
He desires tailored strategies to cultivate healthier eating habits while still enjoying occasional indulgences in delicious cuisine.

## Problem Statement

In Singapore, the increasing prevalence of chronic diseases presents a pressing public health concern, underscoring the need for proactive intervention strategies. 
How can we identify individuals at high risk for chronic diseases based on their behavioral habits? By doing so, we can enable early detection and provide recommendations, fostering a proactive approach to preventing various chronic diseases.

## Data Source (Predictor)

- **Source:** Data sourced from the Behavioral Risk Factor Surveillance System (BRFSS), as detailed on the [CDC's BRFSS Questionnaires page](https://www.cdc.gov/brfss/questionnaires/index.htm).

We chose this dataset as the inputs are comprehensive and of a substantial volume (Combing both 2015 and 2013, we have managed to get more than 10k datapoints for our model training). It is important to note that we have only included data of people with Asian race profile to be more relevant to Singapore. 

## Data Source (Recommender)

The categories and recommended nutrition food profiles are derived from the below webpages:
- [HealthHub Dietary Allowances](https://www.healthhub.sg/live-healthy/recommended_dietary_allowances)
- [HealthHub Calorie Calculator](https://www.healthhub.sg/programmes/nutrition-hub/tools-and-resources#calorie-calculator)
- [HealthHub Protein Importance](https://www.healthhub.sg/live-healthy/why_protein_is_important#:~:text=For%20average%20Singaporean%20adults%20aged,1.2g%2Fkg%20bodyweight%20instead.)
- [HealthHub Getting the Fats Right](https://www.healthhub.sg/live-healthy/getting%20the%20fats%20right#:~:text=Fat%20should%20make%20up%20about,if%20one%20is%20not%20mindful.)
- [USDA National Agricultural Library](https://www.nal.usda.gov/programs/fnic#:~:text=How%20many%20calories%20are%20in,Facts%20label%20on%20food%20packages.)
- [Centrum Singapore - Healthy Diet](https://www.centrum.sg/expert-corner/health-blog/healthy-diet-do-you-follow-dietary-guidelines/)
- [HPB National Nutrition Survey 2022 Report](https://www.hpb.gov.sg/docs/default-source/pdf/nns-2022-report.pdf)
- [Signos - Sugar Intake for Type 2 Diabetics](https://www.signos.com/blog/how-much-sugar-should-a-type-2-diabetic-have-a-day)
- [HealthXchange - Diabetes Glycaemic Index](https://www.healthxchange.sg/diabetes/essential-guide-diabetes/diabetes-glycaemic-index-know)
- [NDTV Food - Dividing Calories in Each Meal](https://food.ndtv.com/food-drinks/how-to-divide-calories-in-each-meal-we-help-deconstruct-it-for-you-1750305#:~:text=NIN%20recommends%20dividing%20equal%20portion,the%20total%20calories%20you%20consume.)
- [Statistics Canada - Sodium Intake](https://www150.statcan.gc.ca/n1/pub/82-003-x/2006004/article/sodium/4148995-eng.htm)

The nutritional profile of the dishes are labelled into their cuisine types manually, and the nutrition values can either be found in the below link in [ObservableHQ - SG Hawker Food Nutrition](https://observablehq.com/@yizhe-ang/sg-hawker-food-nutrition) or manually scrapped as per `01_Data_Collection.ipynb` from [HPB website](https://focos.hpb.gov.sg/eservices/ENCF/)


### Classification Model and Evaluation

We have attempted to hypertuned two models:
1. Logistic Regression
2. Support Vector Classification


With our scores as follow: 

F1 Scores:

| Model                   | Train Score | Test Score | Difference  |
|-------------------------|-------------|------------|-------------|
| Logistic Regression     | 0.601251    | 0.607221   | -0.005970   |
| Support Vector Classifier | 0.595136    | 0.597097   | -0.001961   |

Accuracy: 

| Model                   | Train Score | Test Score | Difference  |
|-------------------------|-------------|------------|-------------|
| Logistic Regression     | 0.718       | 0.719652   | -0.001652   |
| Support Vector Classifier | 0.683381    | 0.681155   | 0.002226    |

Recall (Sensitivity):

| Model                   | Train Score | Test Score | Difference  |
|-------------------------|-------------|------------|-------------|
| Logistic Regression     | 0.625628    | 0.637443   | -0.011815   |
| Support Vector Classifier | 0.684788    | 0.694977   | -0.010189   |

Precision: 

| Model                   | Train Score | Test Score | Difference  |
|-------------------------|-------------|------------|-------------|
| Logistic Regression     | 0.578703    | 0.579734   | -0.001031   |
| Support Vector Classifier | 0.526242    | 0.523384   | 0.002858    |


Both of the hyperturned models do not show significant improvement from the baseline model. However, we see that the hypertuned models performed better in term of generalization as the gap between train and test scores are reduced. We chose Logistic Regression in the end due to the higher F1 and accuracy. 

We then did an analysis on items were classified wrongly by our model. They were mainly old age people with high blood pressure and high cholesterol, while they are not diagnosed with any chronic disease in our datasets, it makes empirical sense that these should be highlighted as high risk by our model. The next notebook will be with regards to the implementation of our recommendation model and this notebook will conclude the part on modelling. 

### Recommender development and Evaluation

The team developed a content-based recommender system for dietary planning, currently in beta for our Streamlit showcase. This system is designed to offer meal suggestions tailored to users' dietary needs and health profiles. It operates by analyzing user input regarding health conditions, activity levels, and dietary preferences to suggest meals that align with their nutritional requirements.

We observed several shortcoming for our recommender as per below: 
  - Overlap in meal suggestions across different profiles.
  - Recommendations could better match individual health and dietary specifics.

The Streamlit app can be accessible from this URL - https://healthapp-chroniscope-demo.streamlit.app/

### Cost Benefit Analysis

1. **Enhanced Quality of Life:** Our project focuses on empowering Singaporeans to achieve a better quality of life by addressing chronic diseases that cause disability, thereby reducing barriers to workforce participation.

2. **Economic Advantages:** Through our initiatives, we anticipate significant economic benefits for Singapore. According to a study conducted by [Annals, Academy of Medicine, Singapore](https://annals.edu.sg/healthcare-cost-of-patients-with-multiple-chronic-diseases-in-singapore-public-primary-care-setting/#:~:text=It%20has%20been%20reported%20that,public%20primary%20care%20was%20SGD303), the projected cost savings per capita to the nation due to our interventions is estimated to be SGD 15,148. This substantial cost reduction not only benefits individuals but also contributes to the overall economic well-being of the nation.

3. **Healthcare Resource Optimization:** By reducing the burden on health resources, our efforts aim to ensure a more sustainable healthcare system in Singapore, alleviating strain and enhancing efficiency for the benefit of all citizens.

These three pillars underscore our commitment to fostering social progress and well-being in Singapore through our project endeavors.
  
We also did a projection for our app as per below: 

**Cost Assumptions**
| Label | Source | 2024       | 2025       | 2026       | 2027       |
|--------|-------|------------|------------|------------|------------|
| Data gathering effort (model training) | [National Population Health Survey 2022](https://www.moh.gov.sg/resources-statistics/reports/national-population-health-survey-2022) | $50,000.00 | $20,000.00 | $20,000.00 | $20,000.00 |
| Data gathering effort (recommender)    | [HPB eServices](https://focos.hpb.gov.sg/eservices/ENCF/) | $50,000.00 | $10,000.00 | $10,000.00 | $10,000.00 |
| App development (inclusive of backend works) | [Couchbase Blog](https://www.couchbase.com/blog/app-development-costs/) | $500,000.00 | $51,750.00 | $53,561.25 | $55,435.89 |
| App maintenance | [MyCareersFuture](https://www.mycareersfuture.gov.sg/job/engineering/full-stack-engineer-seventh-sense-artificial-intelligence-6978f0b40b39bfd80bf5fa9540dfb55c?source=MCF&event=Search) | $129,600.00 | $134,136.00 | $138,830.76 | $143,689.84 |
| Marketing cost (per user level)        | [PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6416217/) | $70.00    | $72.45    | $74.99    | $77.61    |

**Benefit Assumptions**

| Label | Source | 2024  | 2025  | 2026  | 2027  |
|--------|-------|-------|-------|-------|-------|
| % of chronic disease in Singapore (multimorbidity) | [PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6902794/#:~:text=With%2016.3%25%20of%20the%20Singapore) | 16.30% | 16.30% | 16.30% | 16.30% |
| Users on Healthhub (based on 2017)                  | [Tech.gov.sg](https://www.tech.gov.sg/media/technews/a-healthy-hub-at-your-fingertips) | 84,000 | 88,200 | 92,610 | 97,241 |
| % Users on Healthhub with multimorbidity            | [Tech.gov.sg](https://www.tech.gov.sg/media/technews/a-healthy-hub-at-your-fingertips) | 13,692 | 14,377 | 15,095 | 15,850 |

**Cost-Benefit Analysis**

| % Conversion | Projected Cost Savings |
|--------------|------------------------|
| 0.50%        | $119,578               |
| 0.60%        | $289,413               |
| 0.70%        | $459,249               |
| 0.80%        | $629,085               |
| 0.90%        | $798,920               | 
| 1%           | $968,756               |
| 2%           | $2,667,111             |
| 3%           | $4,365,467             |
| ...          | ...                    |  


**Remarks**
- Aggressive with costs, conservative with benefits to avoid unnecessary commitment of resources.

## Assumptions and Limitations

There are limitations to both of our classifier and recommender.

For classifier, we achieved an accuracy score of 72% for test results. While this is higher than a purely simplistic model where they assume everyone belongs to the majority class, we acknowledge that:
1. There are factors that are not within lifestyle data that is not captured within our training data such as genetic reasons.
2. Our training data stems from the US afterall. Even though we have filtered out to only using Asian data, we need to acknowledge the fact that there are fundamentally difference in the external factors affect an individual in Singapore vs in the US 

For our recommender, we managed to prove the implementation but we are definitely no food scientist and the topic is more complex than that. It is important to note that we made the assumption that the caloric requirements is split evenly across three meals in a day and also the recommendation provided covers the entire meal for a person without additional side dishes or drinks. We will also highlight again that our recommender is in a beta stage and it should not be used to provide any definitive recommendations apart from the showcase. 

## Conclusion

All in all, the team has managed to develop a classifier and a proof of concept for the recommendation. There are clear next steps that will help to improve the classifer and the recommender such as:
1. Gathering of more granular and specific nutritional data for dish for recommenders' evaluation.
2. Obtaining the Singapore health survey data that will make the context more localized.  


### Appendix: Data Dictionary

| Feature   | Description                                                                                                                                                    | Categories / Values                                                                                                                                  |
|:----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|
| cpd_bronchitis  | (Ever told) you have Chronic Obstructive Pulmonary Disease or COPD, emphysema or chronic bronchitis?                                                           | 0 - No and others<br>1 - Yes                                                                                                                          |
| depression  | (Ever told) you that you have a depressive disorder, including depression, major depression, dysthymia, or minor depression?                                  | 0 - No and others<br>1 - Yes                                                                                                                          |
| arthritis  | Respondents who have had a doctor diagnose them as having some form of arthritis                                                                               | 0 - No<br>1 - Yes                                                                                                                                     |
| heart_attack  | (Ever told) you had a heart attack, also called a myocardial infarction?                                                                                      | 0 - No and others<br>1 - Yes                                                                                                                          |
| stroke  | (Ever told) you had a stroke.                                                                                                                                  | 0 - No and others<br>1 - Yes                                                                                                                          |
| asthma   | (Ever told) you had asthma?                                                                                                                                    | 0 - No and others<br>1 - Yes                                                                                                                          |
| diabetes  | (Ever told) you have diabetes (If "Yes" and respondent is female, ask "Was this only when you were pregnant?". If                                              | 0 - No and others<br>1 - Yes (include pregnancy)                                                                                                      |
| kidney_disease  | (Ever told) you have kidney disease? Do NOT include kidney stones, bladder infection or incontinence.                                                          | 0 - No and others<br>1 - Yes                                                                                                                          |
| heart_disease    | Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI)                                                          | 0 - No<br>1 - Yes                                                                                                                                     |
| skin_cancer  | (Ever told) you had skin cancer?                                                                                                                               | 0 - No<br>1 - Yes                                                                                                                                     |
| other_cancer  | (Ever told) you had any other types of cancer?                                                                                                                 | 0 - No and others<br>1 - Yes                                                                                                                          |
| high_cholesterol   | Adults who have had their cholesterol checked and have been told by a doctor, nurse, or other health professional that it was high                             | 0 - No<br>1 - Yes and others                                                                                                                          |
| high_bp  | Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional                                                       | 0 - No and others<br>1 - Yes                                                                                                                          |
| exercise_cat  | Adults that participated in 150 minutes (or vigorous equivalent minutes) of physical activity per week.                                                       | 0 - 0 minutes of vigorous exercise<br>1 - 1 to 149 minutes of vigorous exercise<br>2 - more than 150 minutes of vigorous exercise                     |
| one_alc_per_day  | Adults who reported having had at least one drink of alcohol in the past 30 days.                                                                              | 0 - No and others<br>1 - Yes                                                                                                                          |
| blind     | Are you blind or do you have serious difficulty seeing, even when wearing glasses?                                                                             | 0 - No<br>1 - Yes                                                                                                                                     |
| martial   | Are you: (marital status)                                                                                                                                      | 0 - Not married + others<br>1 - Married                                                                                                               |
| binge_drink  | Binge drinkers (males having five or more drinks on one occasion, females having four or more drinks on one occasion)                                          | 0 - No<br>1 - Yes                                                                                                                                     |
| ave_drink_week  | Calculated total number of alcoholic beverages consumed per week                                                                                               | Continuous (Integer)                                                                                                                                  |
| fruit   | Consume Fruit 1 or more times per day                                                                                                                          | 0 - consumed fruits less than one time per day<br>1 - consume fruits more than one time per day                                                       |
| vegetable   | Consume Vegetables 1 or more times per day                                                                                                                     | 0 - consumed veg less than one time per day<br>1 - consume veg more than one time per day                                                             |
| diff_walking  | Do you have serious difficulty walking or climbing stairs?                                                                                                     | 0 - No<br>1 - Yes                                                                                                                                     |
| occasion_drink_30days   | During the past 30 days, how many days per week or per month did you have at least one drink of any alcoholic beverage such as beer, wine, a malt beverage?  | Continuous (Integer)                                                                                                                                  |
| employment_status   | Employment status                                                                                                                                              | 0 - unemployed + undisclosed<br>1 - employed                                                                                                          |
| smoker_status  | Four-level smoker status: Everyday smoker, Someday smoker, Former smoker, Non-smoker                                                                           | 0 - never smoke<br>1 - former smoker<br>2 - current smoker (smoke some days)<br>3 - current smoker (smoke every day)                                   |
| sex       | Indicate sex of respondent.                                                                                                                                    | 0 - Female<br>1 - Male                                                                                                                                |
| education   | Level of education completed                                                                                                                                   | 0 - Did not graduate high school<br>1 - high school graduate<br>2 - College or Tech school grad                                                       |
| race     | Race/ethnicity categories                                                                                                                                      | 1 - White only, Non-Hispanic<br>2 - Black only, Non-Hispanic<br>3 - American Indian or Alaskan Native only, Non-Hispanic<br>4 - Asian Only, Non-Hispanic<br>5 - Native Hawaiian or other Pacific Islander only, Non-Hispanic<br>6 - Other race only, non-Hispanic<br>7 - Multi-racial, non-Hispanic<br>8 - Hispanic<br>9 - Don't know / Not sure / Refused |
| height      | Reported height in meters                                                                                                                                      | Continuous (Float)                                                                                                                                    |
| weight     | Reported weight in kilograms                                                                                                                                   | Continuous (Float)                                                                                                                                    |
| age  | Fourteen-level age category                                                                                                                                    | 1 - 18 to 24<br>2 - 25 to 29<br>3 - 30 to 34<br>4 - 35 to 39<br>5 - 40 to 44<br>6 - 45 to 49<br>7 - 50 to 54<br>8 - 55 to 59<br>9 - 60 to 64<br>10 - 65 to 69<br>11 - 70 to 74<br>12 - 75 to 79<br>13 - 80 or older |




