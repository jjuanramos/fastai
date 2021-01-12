1. Can we always use a random sample for a validation set? Why or why not?

No. For time series, it is better to use as validation set the more recent data (excluding the test set), as the purpose of the algorithm is to predict the future.

2. What is overfitting? Provide an example.

Overfitting consists of the algorithm adapting itself too much to the training set and as a consequence losing the ability to generalize on new data.

3. What is a metric? How does it differ from "loss"?

A metric is a value whose purpose is to allow humans to understand how the algorithm performs. On the other hand, a loss is the optimization objective set by the algorithm; it will modify its architectural parameters in orders to optimize the loss.

4. How can pretrained models help?

Pretrained models are useful for problems where data or resources are not abundant. They are also useful because they have been trained by experts and thus they can achieve great results without the necessity of having a machine learning scientist in the team. Pretrained models are not apt for tabular data problems.

5. What is the "head" of a model?

The head of a model is the last layer it has. It is the layer that adapts the model to the problem at hand. Thus, for pretrained models this head is usually cut off and retrained with the data for the new problem.

6. What kinds of features do the early layers of a CNN find? How about the later layers?

The early layers usually find gradients or basic forms. The images found by the layer get more complex after each layer, and thus the later layers are able to detect really complex images.

7. Are image models only useful for photos?

No, image models are useful for everything the human eye can be competent at.

8. What is an "architecture"?

An architecture is the template for a mathematical function.

9. What is segmentation?

The process of creating a model that is able to identify every single pixel in an image. See: self-driving cars need segmentation to 'see' what's in the road.

10. What is y_range used for? When do we need it?

y_range is used in regression algorithms to indicate the range at which the objective value lies.

11. What are "hyperparameters"?

Hyperparameters are the parameters of the parameters. They are used to alter how the model is created, how the parameters are obtained, etc.

12. What's the best way to avoid failures when using AI in an organization?

Keeping a untouched test set until the end.

13. What is a p value?

A p value is the probability of the numbers obtained being a result of chance.

14. What is a prior?



15. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.



16. Where do text models currently have a major deficiency?

At generating artificial, credible conversations or responses.

17. What are possible negative societal implications of text generation models?

Identity suplantation.

18. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?

Embedding the model in a process in which it interacts closely with a human user.

19. What kind of tabular data is deep learning particularly good at?

Recommendation systems.

20. What's a key downside of directly using a deep learning model for recommendation systems?

They tell you what the user might like, instead of what could be useful to that user.

21. What are the steps of the Drivetrain Approach?

 1. Identify the objective you want to achieve.
 2. Identify what actions you could take to achieve that objective.
 3. Identify the data you have (or can acquire) that could help.
 4. Build a model that you can use to determine the best actions to take to get the best results in terms of your objective.

22. How do the steps of the Drivetrain Approach map to a recommendation system?



23. Create an image recognition model using data you curate, and deploy it on the web.



24. What is DataLoaders?



25. What four things do we need to tell fastai to create DataLoaders?



26. What does the splitter parameter to DataBlock do?



27. How do we ensure a random split always gives the same validation set?

By using a random seed.

