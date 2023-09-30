# Supervised Learning Challenge
> How can we use algorithms to process complex data and forecast the future?

## Folder Contents
- A `Resources` folder containing all the data we'll be using for this exercise.
- A `Visuals` folder containing most of the Bokeh charts that can't be previewed on Github (reproduced here in the ReadMe instead)
- A `.gitignore` file that ignores common things like PyCache, Jupyter Notebook checkpoints, and other common gitignorable Python entities. 
- A main `Crypto_Clustering` Jupyter Notebook file that uses the Scikit-Learn module to let us learn about any hidden patterns in our data.
- This `README.md` file will serve as the Credit Risk Analysis Report about the data used.

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

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

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
```

## Resources that helped a lot
We aren't coding any of the machine learning algorithms from scratch. There's no need to reinvent the wheel or rediscover calculus for the purposes of this exercise. However, it's still important to learn about how the algorithms work and when these can be applied. I found these theory videos to be very useful:
- Cassie Kozyrkov's [Making Friends with Machine Learning] 6-hour course is also great for giving people a look into the black boxes that now govern our data-centric world.

## FINAL NOTES
> Project completed on September 7, 2023
