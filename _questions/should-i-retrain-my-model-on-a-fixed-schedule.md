---
title: "Should I retrain my model on a fixed schedule or only when drift is detected?"
section: human-behavior
order: 10
description: "For most small models, we recommend frequent scheduled retraining, which is cheap and fixes natural drift, and spending alerting effort on sudden breaks from data or engineering bugs. Thoughts on ML Engineering After a Year of my PhD describes Task MLEs at small companies who retrain hourly, daily, or weekly, and whose model performance recovers when they do. Retraining many xgboost or scikit-learn models costs a few dollars at most."
sources:
  - title: "Thoughts on ML Engineering After a Year of my PhD"
    url: https://www.sh-reya.com/blog/phd-year-one/
asked:
  - https://www.reddit.com/r/mlops/comments/1t7y4j1/how_do_you_actually_catch_when_your_production/
---
