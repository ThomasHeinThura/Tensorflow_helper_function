* This dataset is a bit tricky to do because input data is tokenized and the training of the dataset is likely to overfit easily.
* I found only one similar example in Kaggle and other examples are taken from another API.(I had no luck in testing with that API.)
* In a similar example in Kaggle, he is training with dense layers. But his method takes a lot of time because there are no flattened layers. And I have no luck with regularizing weights and layers. 
* In this dataset, only conv1D and dense models are easy to train for 3-10min.
* The others RNN models that a lot of time 5min for one epoch and take a lot of memory. I think it is because when the dataset input is tokenized and the training starts with a large tensor array and takes heavy input to RNN layers.
* From my point of view, the main reason the model is easy to overfit and val_loss is hard to come down to 0.8 mark is because of the dataset.
* Trying to import from the text dataset or reverse the tokenized data to the text data and embedded again may fix the problem. 
