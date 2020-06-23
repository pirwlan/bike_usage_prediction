# Predicting bike sharing demand

A couple years ago, Kaggle posted a [competition](https://www.kaggle.com/c/bike-sharing-demand/overview/description "Link to Kaggle competition"), in which the participants were supposed to predict the number of total bike rentals at given time points in Washington DC. The rental count were further distinguished in registered and casual lenders. 

I approached the problem by predicting both, registered and casual lenders separately, which significantly improved performance. 

Using a rather simple and robust solution, on the newly implement PoissonRegressor, I could reach the top 5 % in the competition. 

The data was provided by [Capital Bikeshare](https://www.capitalbikeshare.com/system-data)
