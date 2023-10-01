# Supervised Learning Challenge
> How can we use algorithms to process complex data and forecast the future?

## Folder Contents
- A `Resources` folder containing all the data we'll be using for this exercise.
- A `.gitignore` file that ignores common things like PyCache, Jupyter Notebook checkpoints, and other common gitignorable Python entities. 
- A main `credit_risk_classification` Jupyter Notebook file that uses the Scikit-Learn module to let us analyze our data and make predictions.
- This `README.md` file, aside from installation instructions, will serve as the Credit Risk Analysis Report regarding the data used.

### Installation/Prerequisites
- Make sure you can run Python. The development environment I used was set-up with:
```
conda create -n dev python=3.10 anaconda -y
```

#### Imported Modules
- Installing via the conda command given should give you access to most, if not all, of the script's modules locally. However, if you don't have them, be sure to grab yourself the following libraries:
  - [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) and [NumPy](https://numpy.org/install/) for basic data management and manipulation
  - [Scikit-Learn](https://scikit-learn.org/stable/install.html) for the supervised learning algorithms we'll be using
  - [Imbalanced-Learn](https://imbalanced-learn.org/stable/install.html#getting-started) for a resampling algorithm we'll be using

---

# Credit Risk Analysis Report

## Overview of the Analysis

A commercial bank makes money by giving people loans and collecting interest as people repay what they owe. However, as the 2008 financial crisis has shown, it's generally bad for loans to default as banks both lose money they could've earned and capital they could've used for other investment and basic banking activities. With the historical data we were provided with, we wish to create a model that can predict, based on each applicant's financial status, whether a borrower will default on their loans. These predictions may allow the bank to increase the long-term profits made by ensuring all the loan applicants the bank approves are those with a low chance of defaulting on their payments.

Defaults are bad for business, so let's predict who will default and not give them loans.

### Data Definitions

Each row of the `lending_data.csv` file used represents a loan application filed to the bank. Each feature column describes something about the applicant's financial status:
- `loan_size`: How much money the applicant borrowed.
- `interest_rate`: The interest rate at the time the loan was active.
- `borrower_income`: The borrower's income.
- `debt_to_income`: The ratio of the borrower's total debt to their income.
- `num_of_accounts`: The number of accounts the borrower has.
- `derogatory_marks`: The number of negative items such as bankruptcies and credit risks on the borrower's credit report.
- `total_debt`: The borrower's total debt to be paid.

With data regarding those feature columns, we wish to predict `loan_status`, a binary variable.
- A value of `0` means that the loan was paid as expected. If we predict this, we're assuming that the loan is healthy and won't default.
- A value of `1` means that the borrower defaulted on their loan. If we predict this, we're assuming that the loan will or has a high risk of defaulting.

In our dataset, about 3% of all loans offered have defaulted. We have roughly 75,000 normal loans and 2,500 defaulted loans. Ideally, we'd like the number of defaults to be zero.

### Machine Learning Process

Because the data is heavily imbalanced at about a 30:1 ratio, we constructed two models with the following steps:

**Standard Logistic Regression** is used as the output we're predicting has a binary range. SKLearn will try to draw a logistic curve and plot data points on it to see which binary label they're more likely to fall under.
1. Get the data
2. Split the data into features `X` and labels `y`
3. Split the data into training and testing sets
4. Initialize the Logistic Regression model
5. Fit the model to the training data
6. Predict what the labels would be using `X` testing data
7. Compare how far off the predictions are from our actual `y` testing answers

**Randomly Over-Sampled Logistic Regression** is used as our data is rather imbalanced. By randomly resampling existing data so that our training data has the same number of data points in each label, we could get a better result.
1. Initialize a RandomOverSampler
2. Randomly over-sample our training data
3. Do Logistic Regression just like steps 4-7 of the previous model, but with the model fitted to our resampled data instead
4. Compare how far off the predictions are from our actual `y` testing answers, given our different model used

## Results

The following performance metrics are used:
- [Balanced Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html): The average recall of each class.
- Precision: The number of correctly made calls out of all calls that *predicted* a certain Class.
  - ${Precision} = \frac{True(Class)}{True(Class)+False(Class)}$
  - Class `0` Precision describes how many approved loans are offered to the right people. However, because of class imbalance, this is not the most informative statistic to use. 99% Class `0` Precision on 20,000 predictions means that 200 loans will probably default.
  - Class `1` Precision describes how many denied loans are denied from those who actually would default and don't deserve the chance.
- Recall: The number of correctly made calls out of all calls that *actually* belong to a certain Class.
  - ${Recall} = \frac{True(Class)}{True(Class)+False(Other Class)}$
  - Class `0` Recall describes how many applicants who would repay their loans just fine would have their loans approved.
  - Class `1` Recall describes how many applicants who would default on their loans would have their loans denied.
 
This bank's job is to reduce giving loans to those who don't deserve it, so *Class `1` Recall should be prioritized*. It's more harmful to let a defaulter have their loan than it is to accidentally deny a perfectly healthy application and have the aspiring borrower reapply at a later date.
 
**Standard Logistic Regression**
  - Balanced Accuracy Score: 95.2%
  - Precision:
    - Class `0`: ~100%
    - Class `1`: 85%
  - Recall:
    - Class `0`: 99%
    - Class `1`: 91%

**Randomly Over-Sampled Logistic Regression**
  - Balanced Accuracy Score: 99.4%
  - Precision:
    - Class `0`: ~100%
    - Class `1`: 84%
  - Recall:
    - Class `0`: 99%
    - Class `1`: **99%**

## Summary

As we defined success to mean "offering future defaulters less loans", the model using Randomly Over-Sampled Logistic Regression is the better one here with a Class `1` Recall of almost 100% - Only 4 defaulters were mistakenly allowed loans with this regression model. 
> The other model's 91% Class `1` Recall meant that almost 10% of all future defaults would have their applications approved. This is not conducive to the bank's financial wellbeing.

Though it's unfortunate that, regardless of the model, 15% of those who are denied are incorrectly denied, this is an acceptable tradeoff to make as those applicants can simply reapply at a later date and try their luck then.

---
## Code Breakdown
The premise of this challenge is simple: Given past data, predict future occurrences. Overall, the machine learning code used will look like this:
```python
# Initialize a machine learning model
model = someAlgorithm(parameter1 = x1, parameter2 = x2, ...)

# Fit the model to your data
model.fit(yourDataHere)

# Predict unknown results based on your trained model
predictions = model.predict(yourDataHere)

# Assess performance by comparing test results with reality
performance = somePerformanceMetric(y_reality, predictions)
```

## Resources that helped a lot
We aren't coding any of the machine learning algorithms from scratch. There's no need to reinvent the wheel or rediscover calculus for the purposes of this exercise. However, it's still important to learn about how the algorithms work and when these can be applied. I found these theory videos to be very useful:
- Cassie Kozyrkov's [Making Friends with Machine Learning](https://www.youtube.com/watch?v=1vkb7BCMQd0) 6-hour course is also great for giving people a look into the black boxes that now govern our data-centric world.
- Aside from looking up code blocks on Google or StackOverflow, it's also generally more intuitive to follow the official documentation on [SKLearn](https://scikit-learn.org/stable/user_guide.html#user-guide).

## FINAL NOTES
> Project completed on September 7, 2023
