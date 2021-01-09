# Lesson 1

1. Do you need these for deep learning?

   - Lots of math  F
   - Lots of data  F
   - Lots of expensive computers  F
   - A PhD  F
   
2. Name five areas where deep learning is now the best in the world.

 - Web Search
 - Product Recommendations
 - Text to Speech
 - Face Recognition
 - Finding anomalies in radiology images 

3. What was the name of the first device that was based on the principle of the artificial neuron?

The first device was called the Mark I Perceptron.

4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?

 - A set of processing units
 - A state of activation
 - An output for each unit
 - A pattern of connectivity among units
 - A propagation rule for propagating patterns of activities through the network of connectivities
 - An activation rule for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit (loss function?)
 - A learning rule whereby patterns of connectivity are modified by experience
 - An environment within which the system must operate

5. What were the two theoretical misunderstandings that held back the field of neural networks?

 - That a single layer of a perceptron was unable to learn some simple but critical mathematical functions (such as XOR). (True, but extra layers could be added).
 - That just one extra layer would allow any mathematical function to be approximated. (Again true, but in practice you need more layers of neurons, and thus it did not work as well as expected, + too slow and big to be useful, given that time's hardware capabilities among other things).

6. What is a GPU?

A special kind of processor that can handle thousands of single tasks at the same time.

7. Why is it hard to use a traditional computer program to recognize images in a photo?

Because we would need to write the exact steps needed to achieve the prediction. As images are based on pixels, we would have to find every single of combination of pixels that can turn into the particular image we want to predict, and then hard code it.

8. What did Samuel mean by "weight assignment"?

Weight assignment is a particular choice of values for a set of variables (weights). Given a constant input, modifying the weight assignment will result in a complete different output.

9. What term do we normally use in deep learning for what Samuel called "weights"?

We call it parameters nowadays.

10. Draw a picture that summarizes Samuel's view of a machine learning model.

Input --> Model with Parameters --> Output --> Loss Function is optimized? (we compare predicted values with the labels)

while not optimized:
  Loss Function Feedback --> Parameters are modified --> Output --> Loss Function is optimized?
  

11. Why is it hard to understand why a deep learning model makes a particular prediction?

Because the architecture by which the loss function is optimized will be composed of a set of parameters whose logic is not comprehensible by any human. For example, an image recognition algorithm could be composed of a set of numerical parameters that range from 0 to 1, where 1 is black and 0 is white. When we have 2 parameters it's ok but, what about having 1000?

12. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?

Universal approximation theoren.

13. What do you need in order to train a model?

Labeled data.

14. How could a feedback loop impact the rollout of a predictive policing model?

If the labeled data given to train the model is already biased, then the model will output biased predictions, that if followed will make the data more biased which in turn will make the model even more biased. That is a positive feedback loop, where the more the model is used the more biased the data becomes.

15. Do we always have to use 224×224-pixel images with the cat recognition model?

Old pretrained models required that size, but not newer ones. In fact, if you increase the pixel size you can get a model with better results since it will be able to focus on more details. The trade-off is it will be more expensive as well as slower.

16. What is the difference between classification and regression?

Classification models predict categories or classes, while regression models predict numerical quantities.

17. What is a validation set? What is a test set? Why do we need them?

A validation set is a percentage of the dataset that is not used for training the model, and it's instead used to check how well the trained model performs. 

Testing set is only used once we believe we have obtained the optimal architecture for the given problem, and its purpose is to assess whether or not the architecture is really optimal, or if it has actually overfitted the validation set.

18. What will fastai do if you don't provide a validation set?

It will create one (by default 20% of the training set).

19. Can we always use a random sample for a validation set? Why or why not?

No, it's better to implement a random seed. This way, we know that changes in the accuracy are due to changes in the model and not in the validation set.

20. What is overfitting? Provide an example.

Overfitting is the process by which a model adapts itself too much to a given dataset, losing the ability to predict in unknown datasets, that is, of generalizing.

If we trained our model for a thousand layers with the same validation set, it will probably overfit its parameters to better perform in the validation set. We could probably prove it once we predict the test set, and the accuracy declines.

21. What is a metric? How does it differ from "loss"?

A metric is a function that measures the quality of the model's predictions using the validation set (accuracy, for example). The difference with loss is that loss is created for the training system to update its weights automatically, while the metric is used for human consumption. I.e. better understanding the model's performance.

22. How can pretrained models help?

Pretrained models can help because they have already been trained -duh- in a given dataset by experts. This is useful because it's already very capable out of the box, which for practical purposes is ideal as we will need less data, less time and less money to obtain a useful model.

Using a pretrained model for a task different to what it was originally trained for is called 'transfer learning', and it can be do by fine-tuning its head.

23. What is the "head" of a model?

The head of the model is its last layer, since it is the one that is specifically customized to the original training task (when talking about pretrained models). So, when using a pretrained model, the head should be removed to be replaced with one or more new layers with randomized weights, to fit the dataset we are working with.

24. What kinds of features do the early layers of a CNN find? How about the later layers?



25. Are image models only useful for photos?

No, because many things can be depicted as images, such as sounds (spectrograms), time series, etc.

There could be a good rule of thumb for converting a dataset for an image representation: if the human eye can recognize categories from the images, then a deep learning model should be able to do so too.

26. What is an "architecture"?

The architecture is the template for a mathematical function. It doesn't do anything until we provider values for the parameters it contains.

27. What is segmentation?

The process of creating a model that is able to recognize the content of every individual pixel in an image.

28. What is `y_range` used for? When do we need it?

It's used when predicting a numerical, continuous value, and it sets the range our target has.

29. What are "hyperparameters"?

They are parameters about parameters, since they are higher-level-choices that govern the meaning of the weight parameters. They are related to the tweaking the model's creation. Examples would be network architecture, learning rates, data augmentation strategies, etc.

30. What's the best way to avoid failures when using AI in an organization?

Having a test set that keeps hidden from the modelization process so that we can make sure that we are not overfitting the algorithm.

31. Why is a GPU useful for deep learning? How is a CPU different, and why is it less effective for deep learning?

As [this article](https://medium.com/@shachishah.ce/do-we-really-need-gpu-for-deep-learning-47042c02efe2) puts it, GPU is like a truck, while CPU is like a ferrari. The CPU can fetch small amounts of memory into the RAM faster (lower latency), while the GPU takes longer to fetch that memory into the ram, but fetches a bigger amount of memory (greater memory bandwith). The slowness of GPU can be attacked through thread parallelism (use more trucks simultaneously).

Thus, GPU is better to train Deep Learning models, as they are associated with 1. lots of data and 2. iterative operations with that data (such as loss functions).

32. Try to think of three areas where feedback loops might impact the use of machine learning. See if you can find documented examples of that happening in practice.

Articles talking about feedback loops:

 - [Article 1](https://towardsdatascience.com/dangerous-feedback-loops-in-ml-e9394f2e8f43)
 - [Video](https://www.coursera.org/lecture/optimize-machine-learning-model-performance/positive-feedback-loops-negative-feedback-loops-10zPL)

