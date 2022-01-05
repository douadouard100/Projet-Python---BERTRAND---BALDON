Valentine BALDON - Edouard BERTRAND - DIA1

Our dataset is called "Facebook Comment Volume Dataset" and can be found at this link :
https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset

The task at hand was to use the dataset variables to predict the number of comments a Facebook post would
receive in the next H hours.

We found that the most impacting variables to determine this data are the number of comments on the post
added in the last 24 hours and 24 hours before that, the number of comments 24 hours after publication
and the total number of comments at the time of prediction.
We found that some category such as TV shows or comedy favorize getting higher number of comments on a
post, as well as some days of the week, the best one being tuesday.

The best model we found to estimate the proportion of new comments is a Gradient Boosting model that
allowed us to reach ~63% accuracy on our predictions.

In order to use the flask application:
Run the app script and connect to the link that appears in the console. When entering your data, you can
check the number associated to each category in the "categories.txt" file in the flask folder.

Visualizations of the data can be found in the jupyter notebook.
