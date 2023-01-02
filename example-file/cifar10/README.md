These are models that train on cifar10 dataset.
Short Note from testing and building model
1. val_accuary is just a number that looks great to confirm these numbers are accurate then first look at val_loss values if val_loss is over 1 or 0.8 that means the Model is likely to overfit. your prediction values are likely to get lower values. 
2. always calculates f1 and accuracy scores to confirm your models are not overfit. Try not to satisfy with just the val_accuary score being high.
3. dropout and regulation of the model are as important as preparing the data. 