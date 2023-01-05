|models                   |val_accuary    |val_loss       |time       |epochs         |
|-------------------------|---------------|---------------|-----------|---------------|
|imdb_hub_conv            | 87.22%        |0.3542         |2min54sec  |10             |
|imdb_hub_dense           | 87.02%        |0.3323         |2min32sec  |10             |
|imdb_USE                 | 85%           |0.3274         |3min38sec  |10             |
|imdb_hub_lstm            | 83%           |0.5017         |2min43sec  |10             |
|imdb_conv                | 86.11%        |0.3410         |1min40sec  |10             |
|Navie bayes              | 85.89%        |???            |???        |??             |


* This dataset is a bit tricky to do because input data is tokenized and the training of the dataset is likely to overfit easily.
* I found only one similar example in Kaggle and other examples are taken from another API.(I had no luck in testing with that API.)
* In a similar example in Kaggle, he is training with dense layers. But his method takes a lot of time because there are no flattened layers. And I have no luck with regularizing weights and layers. 
* In this dataset, only conv1D and dense models are easy to train for 3-10min.
* The others RNN models that a lot of time 5min for one epoch and take a lot of memory. I think it is because when the dataset input is tokenized and the training starts with a large tensor array and takes heavy input to RNN layers.
* From my point of view, the main reason the model is easy to overfit and val_loss is hard to come down to 0.8 mark is because of the dataset.
* Trying to import from the text dataset or reverse the tokenized data to the text data and embedded again may fix the problem. 

""" hub
The model performace : 
val_accuracy : 79.79%
val_loss : 0.9912
time : 5min50sec
f1 : 0.7935
epochs : 10
"""

""" USE
The model performace : 
val_accuracy : 80.45%
val_loss : 0.8013
time : 1min45sec
f1 : 0.7973
epochs : 15
"""

""" conv1D
The model performace : I cheat alot in this dataset
val_accuracy : 81% 
val_loss : 0.9033
Time : 3min20sec
"""

"""
The model performace : I cheat alot in this dataset
val_accuracy : 81% 
val_loss : 0.8064
Time : 30sec
"""

"""
The model performace : 
val_accuracy : 59.97%
""