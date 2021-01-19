from fastai.vision.all import *
from fastbook import *

# 1. Set the path

path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path

# 2. Load the images

threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()

valid_threes = (path/'valid'/'3').ls().sorted()
valid_sevens = (path/'valid'/'7').ls().sorted()

# 3. Transform every image into a tensor and put them all in a list

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]

valid_seven_tensors = [tensor(Image.open(o)) for o in valid_sevens]
valid_three_tensors = [tensor(Image.open(o)) for o in valid_threes]

# 4. Transform the list of tensors into a tensor of tensors, and divide every value of every tensor by 255 (so the value ranges between 0 and 1)

stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255

valid_stacked_sevens = torch.stack(valid_seven_tensors).float()/255
valid_stacked_threes = torch.stack(valid_three_tensors).float()/255

stacked_sevens.shape #shape
stacked_sevens.ndim #rank

# 5. Concat the tensors into one and transform the tensor from rank 3 to rank 2 (converts the matrix to a row vector)

train_x = torch.cat([stacked_sevens, stacked_threes]).view(-1, 28*28)
valid_x = torch.cat([valid_stacked_sevens, valid_stacked_threes]).view(-1, 28*28)

# 6. Creates a tensor of 0 (as label for sevens) and 1 (as label for threes), and returns a column tensor

train_y = tensor([0]*len(sevens) + [1]*len(threes)).unsqueeze(1)
# unsqueeze returns a tensor of dimension one
# put in a specific position. 0 == one row, 1 == one column
valid_y = tensor([0]*len(valid_sevens) + [1]*len(valid_threes)).unsqueeze(1)

# 7. Returns a list of tuples (x, y)

dset = list(zip(train_x, train_y))
valid_dset = list(zip(valid_x, valid_y))

# 8. Creates a dataloader composed by mini-batches of 256 items


dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

# 9. Creates a DataLoaders item with training and validation set

dls = DataLoaders(dl, valid_dl)

# 10. Loss function that returns the mean of the sigmoid function for the predictions.
# Modifies the prediction value accordingly with the target value

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()


# 11. Accuracy function that, given a batch, returns the mean number of correct cases for that batch

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

# 12. Learner composed by a dataloaders, a model, an optimization function, a loss function and a metrics function

learn = Learner(
    dls,
    nn.Linear(28*28, 1),
    opt_func=SGD,
    loss_func=mnist_loss,
    metrics=batch_accuracy
)

# 13. Fit the model for 15 epochs with a learning rate of 0.5

learn.fit(n_epoch=15, lr=0.5)


# And that's it! We have a trained linear model able to differentiate 3s from 7s!
