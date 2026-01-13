# PL Predicts


(Project will be up again soon! Making a few UI changes as I wasn't satisfied with it)

## How It's Made
**Tech used:** HTML, CSS, JavaScript, Python, Sci-kit Learn, pandas, Flask

A web app that allows you to search up any player in the league (specifically the English Premier League) and predict how many fantasy points they would accumulate over a 5-game period.


## Lessons Learned:

* **Random Forest Regression**: Implemented ensemble learning methods and tuned hyperparameters using grid search and cross-validation for 85%+ prediction accuracy
* **Data Processing**: Cleaned and preprocessed raw FPL datasets, handling missing values and selecting relevant features from historical statistics
* **Feature Engineering**: Created derived features like rolling averages, form indicators, and fixture difficulty from season records to improve model performance
* **Full-Stack Integration**: Built end-to-end pipeline connecting Flask API backend with React frontend for real-time player performance forecasting
