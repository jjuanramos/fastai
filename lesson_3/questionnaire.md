1. How is a grayscale image represented on a computer? How about a color image?

A grayscale image is represented as a matrix of size height x width where every value in the matrix is a pixel. Each pixel is an integer that ranges from 0 (white) to 255 (black), with shades of gray between the two.

Color images are also represented as a matrix of size height x width, where every value in the matrix is a pixel. This time though, the value of the pixel is not an integer, but a vector composed of three different values where each represent the amount of red, green and blue in the image. [Source](https://homepages.inf.ed.ac.uk/rbf/HIPR2/value.htm#:~:text=For%20a%20grayscale%20images%2C%20the,is%20taken%20to%20be%20white.)

2. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?

.
├── item_list.txt                  
├── labels.csv           
├── history.csv              
├── models                   
├── train                    
|   ├── 7          
|   └── 3
└── valid                   
    ├── 7
    └── 3
    
The dataset follows a common way for setting up ML models. The training and validation set (and/or test set) have their own folders. Inside those folders, we find a set of subfolders where each subfolder corresponds to one of the labels or targets of the dataset.

3. Explain how the "pixel similarity" approach to classifying digits works.

It consists of obtaining the average value of every pixel for the 3s, and the same for the 7s.
Then, for predicting what number a new case is, we compute the distance between the value for pixel i of the new case and the average value for pixel i in the 3s (for example). We do the same but against the average pixel value of the 7s. We will label our case with the same value where the mean absolute error (or rmse) is less.

4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.

List comprehensions consist of brackets that have an expression followed by an if clause, then zero or more for or if clauses. [Source](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python#:~:text=List%20comprehensions%20provide%20a%20concise,kinds%20of%20objects%20in%20lists.)

```{python}

my_tensor = torch.arange(20)
double_odds_tensor = [2*val for val in my_tensor if val % 2 != 0]

```

5. What is a "rank-3 tensor"?

A 'rank-3 tensor' is a single three dimensional tensor.
This could mean a stacked tensor of rank-2 tensor (matrices). For this image recognition example is what it means, as each image is a rank-2 tensor (a matrix with height (vector --> 1-rank tensor) and width (vector --> 1-rank tensor).

6. What is the difference between tensor rank and shape? How do you get the rank from the shape?

The shape of a tensor is the length / size of each axis. For an image, its tensor shape would be torch.Size([28, 28]), indicating that the height vector has a length of 28 (28 pixels in this case) and the same for the width vector.

Rank indicates the number of axis the tensor has. For the same image it would mean that it is rank-2, as it has axis.

7. What are RMSE and L1 norm?

L1 norm == Mean Absolute Error. It is the average of the absolute difference of each of the predicted values against the target value.
L2 norm == Root Mean Squared Error. It is the average of the root of the square difference of each of the predicted values against the target value.

8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?

By taking advantage of the Numpy arrays and Pytorch tensors, that compile in C.

9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.

```{python}

tns = tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
)

tns = tns * 2
tns.flatten()[-4:]

```

10. What is broadcasting?

Broadcasting is a capability existing in Pytorch tensor class that consists of, when a computation is made between two tensors of different ranks, it will expand the smaller tensor to have the same rank as the greater one.
After broadcasting so the two arguments of the operation have the same rank, Pytorch applies its usual logic for the two tensors of the same rank: it performs the operation on each corresponding element of the two tensors, and returns the tensor result.

11. Are metrics generally calculated using the training set, or the validation set? Why?

Metrics should be calculated using the validation set, as otherwise we could get better metrics than they actually are as a consequence of overfitting.

12. What is SGD?

Stochastic Gradient Descent is the iterative process through which a set of initially randomized parameters are optimized for a given loss function. This process would go like this:

 1. Initialize the parameters of the model randomly.
 2. Predict the label for each set of independent variables.
 3. Calculate the loss function of the prediction against the target.
 4. Apply backpropagation to obtain the gradient of each parameter for the loss function (how changing that parameter by one unit would change the loss).
 5. Modify the parameters against the gradient, with a given learning rate (This is called _step the parameters_).
 6. Reset the gradient variables.
 7. Repeat steps 2-6 for a number of preset epochs, because the metric has achieved your objective or because the metric has started to get worse.

13. Why does SGD use mini-batches?

The process by which the weights are updated with the gradients is called an _optimization step_. In order to obtain these weights, we should decide how much data to use. Using the whole dataset is probably too much, and thus the process of calculating the gradients for the optimization step will take a long time. On the other hand, taking too little data will turn into not getting enough information about our dataset, and thus the gradients won't be of as much quality.

So, we compromise and obtain the gradients for a few data items every time. These few data items are called _mini-batches_. The number of items in a mini-batch is called _batch size_.

14. What are the seven steps in SGD for machine learning?

 1. Randomly initialize the parameters.
 2. Predict the target values for a batch of the training set.
 3. Calculate the loss function for those predictions against the actual values.
 4. Use backpropagation to obtain the gradients of the parameters in that epoch.
 5. Step the parameters using the gradient and a learning rate.
 6. Reinitialize the gradients.
 7. Iterate 2-6 until the number of epochs indicated is completed, the model achieves a target metric or the metric starts getting worse.

15. How do we initialize the weights in a model?

Randomly.

16. What is "loss"?

The loss is the number we are trying to make as small as possible in order to achieve our best predicting results possible.

17. Why can't we always use a high learning rate?

Because a too high learning rate can turn into the loss getting worse or bouncing away from the local / global minimum of the loss function.

18. What is a "gradient"?

A gradient is the partial derivative of the loss function for a particular parameter. In other words, it is how much the loss function varies when a specific parameter changes by one unit.

19. Do you need to know how to calculate gradients yourself?

No, Pytorch / Keras / Tensorflow does it for you.

20. Why can't we use accuracy as a loss function?

Because we need a loss function to calculate our gradient. Again, the gradient indicates us how steep the slope is for a certain value of a certain parameter. The problem with accuracy is that its gradient is 0 almost everywhere, except when it goes from one label to another (when accuracy == 0.5), because it's constant almost everywhere.

21. Draw the sigmoid function. What is special about its shape?

The sigmoid function is a function where for every value of x, y is always between 0 and 1. It makes it easier for the SGD to find meaningful gradients as it is a smooth curve that only goes up.

22. What is the difference between a loss function and a metric?

A loss function is a function given to our optimizer to find the optimal set of weights for a model, while a metric is a formula that allows us to understand how well the model is performing.

23. What is the function to calculate new weights using a learning rate?

weights = weights - learning rate * gradients.

24. What does the `DataLoader` class do?

A Dataloader class takes a Python collection and turns it into an iterator over many batches.

25. Write pseudocode showing the basic steps taken in each epoch for SGD.

```

for epoch in epochs:
    for features, target in batches_dataloader:
        pred = model(features)
        loss = loss_func(pred, target)
        loss.backward()
        parameters = parameters - learning_rate * parameters.grad
        parameters.grad.zero_()

```

26. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?

```

def returnTupleArray(a, b):
    return list(zip(a, b))

```

A _Dataset_ in Pytorch has to return a tuple of (x, y) when indexed. This function gives that functionality.


27. What does `view` do in PyTorch?

It is a method that changes the shape of a tensor without changing its content.

28. What are the "bias" parameters in a neural network? Why do we need them?

The bias parameters act as the intercept for the neural network. It is an additional parameter that is used to adjust the output of the model. It is a constant that helps the model better fit the data.

29. What does the `@` operator do in Python?

It multiplies two matrices.

30. What does the `backward` method do?

The backward function computes the gradient for every parameter of the tensor x with requires_grad=True for a given loss function and adds them to x.grad

31. Why do we have to zero the gradients?

Because the backwards method adds the gradients of the loss function to the currently stored gradient weights.

32. What information do we have to pass to `Learner`?

 1. a DataLoaders object.
 2. a model.
 3. an optimization function. Difference between optimization and loss function is that the optimization function is used to optimize a loss function. SGD is an optimization function, and you could use it to optimize the weights and biases of the model with a loss function such as l1 or l2.
 4. a loss function.
 5. a (optional) metric.

33. Show Python or pseudocode for the basic steps of a training loop.

```

def train_model(model, epochs):
    for epoch in epochs:
        for xb, yb in training_batches_dataloaders:
            preds = model(xb)
            loss = loss_function(preds, yb)
            loss.backward()
            params = params - learning_rate * params.grad
            params.grad.zero_()
            
            check the accuracy for the whole validation dataset: we would obtain the accuracy for each minibatch and then obtain the mean of all minibatches.

```

34. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.

Relu is an activation function that, for values that are below zero returns zero. Else it returns the same input.

35. What is an "activation function"?

A nonlinearity or activation function, is the layer that is used to obtain the outputs of a linear layer and decide whether we should 'activate' each output. Activation functions should be added between each pair of linear layers, otherwise these linear layers stack and it is as if we only had one (big) linear layer.

36. What's the difference between `F.relu` and `nn.ReLU`?

F.relu is a function while nn.ReLU is a module. When using nn.Sequential, Pytorch requires us to use the module version.

37. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?

Because in practice deeper models work better (they are more performant).

