## Basic ML Review

<!-- toc -->


<!-- tocstop -->

> **_NOTE:_** This is a quick refresh of some key concepts touched on in the book. This is not meant to be an introduction to machine learning. For readers who want an introduction to ML, I recommend the following resources:
> 1. [Lecture notes] [Stanford CS 321N](https://cs231n.github.io/): deep learning focused, beginner-friendly.
> 2. [Book] Kevin P Murphy's [Machine Learning: A Probabilistic Perspective](https://probml.github.io/pml-book/book1.html): foundational, comprehensive, though a bit intense.

A model is a function that transforms inputs into outputs, which can then be used to make predictions. For example, a binary text classification model might take sentences as inputs and output values between 0 and 1. You can use these output values to make predictions, such as if the value is less than 0.5, output the `NEGATIVE` class, and if the value is greater than or equal to 0.5, output the `POSITIVE` class.

In traditional programming, functions are given and outputs are calculated from given inputs. For example, your function f(x) might be given as: `f(x) = 2x + 3`.

Given x = 1, the output will be `f(1) = 2 * 1 + 3 = 5`. Given x = 3, the output will be `f(3) = 2 * 3 + 3 = 9`.

In supervised ML, the inputs and outputs are given, which are called data, and the function is derived from data. Given x as input and y as output, you want to learn a function f such that applying f on x will produce y. However, ML isn’t powerful enough to derive arbitrary functions from data yet, so you still need to specify the form that you think the function should take[^1]. It can be a linear function, a decision tree, a feedforward neural network with two hidden layers, each with 768 neurons[^2].

For example, given a dataset with only two examples (x = 1, y = 5) and (x = 3, y = 9), you might specify that the function is a linear function, which means that it takes the form f(x) = wx + b. Then you learn the values of w and b to fit this dataset. Because w and b are learned during the training process, they are called parameters. 

For each type of model, there are many possible values for the parameters. You need an objective function to evaluate how good a given set of parameters is for a dataset, and a procedure to derive the set of parameters best suited for the given data according to that objective, known as a learning procedure.

> **_SIDEBAR:_** Some readers might wonder if the above paragraph about parameters still applies to non-parametric models such as K-means clustering and decision trees. Being non-parametric doesn’t mean that models don’t use parameters. In a parametric model, the number of parameters is fixed with respect to the sample size. In a nonparametric model, the effective number of parameters can grow with the sample size. So the complexity of the function underlying a neural network remains the same even if the amount of data grows. But the complexity of the function underlying a decision tree grows as its number of nodes grows.

When talking about model selection, most people think about selecting a function form. However, choosing the right objective function and a learning procedure is extremely important in finding a good set of parameters for your model.


### Objective Function

The **objective function**, also known as the loss function, is highly dependent on the model type and whether the labels are available. If the labels aren’t available, as in the case of unsupervised learning, the objective functions depend on the data points themselves. For example, for k-means clustering, the objective function is the variance within data points in the same cluster (so the objective is to put data points into clusters so that the within-cluster variance is minimized). But unsupervised learning is much less commonly used in production.

Most algorithms you’ll encounter in production are supervised or some form of weakly or semi-supervised, as mentioned in the section **[Handling the Lack of Labels](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch04.html)** in Chapter 4. Given a set of parameter values, you calculate the outputs from the given inputs, and compare the given function’s predicted outputs (y') to the actual outputs (y). Objective functions evaluate how good a set of parameter values is by measuring the distance between the set of y' and the set of y.

To make this concrete, let’s go back to the example above where we have only 2 data points (x = 1, y = 5) and (x = 3, y = 9). We want to find w and b such that `f(x) = wx + b` best suited this data. Given the set of parameter values w = 3 and b = 4, we get the predicted outputs of 7 and 13 as shown in Table A-1. The objective function measures the distance between the predicted outputs (7, 13) and the actual outputs (5, 9).


<table>
  <tr>
   <td><strong>Input</strong>
   </td>
   <td><strong>Predicted output made by f(x | w=3, b=4) = 3x + 4</strong>
   </td>
   <td><strong>Actual output</strong>
   </td>
  </tr>
  <tr>
   <td>x = 1
   </td>
   <td>3 * 1 + 4 = 7
   </td>
   <td>5
   </td>
  </tr>
  <tr>
   <td>x = 3
   </td>
   <td>3 * 3 + 4 = 13
   </td>
   <td>9
   </td>
  </tr>
</table>


Table A-1: Predicted outputs when w = 4 and b = 4

There are many types of distance metrics you can use to derive your objective functions. When the outputs are scalars (numbers), two common metrics are Root Mean Squared Error and Mean Absolute Error as shown in Table A-2.


<table>
  <tr>
   <td><strong>Objective function</strong>
   </td>
   <td><strong>How to calculate</strong>
   </td>
   <td><strong>Distance metrics</strong>
   </td>
  </tr>
  <tr>
   <td>Root Mean Squared Error (RMSE)
   </td>
   <td>

<p>
$\sqrt{\sum\limits_{i=1}^n \frac{(y_i' - y_i)^2}{n}}$
   </td>
   <td>Euclidean
   </td>
  </tr>
  <tr>
   <td>Mean Absolute Error (MAE)
   </td>
   <td>

<p>
$\sum\limits_{i=1}^n \frac{|y_i' - y_i|}{n}$
   </td>
   <td>Manhattan
   </td>
  </tr>
</table>

Table A-2: Two common objective functions for scalar outputs

However, many types of models don’t output just one number given an input, but output a distribution. For example, if your task has three classes: [cat, dog, chicken], your model might output an array of how likely it is that your input belongs to each class. So the predicted output might look like [0.1, 0.5, 0.4], which means the input has 10% chance of being cat, 50% chance of being dog, 40% chance of being chicken. The actual label for this example is chicken, so the output is [0, 0, 1]. We want to measure the distance between the predicted outputs that take the form [0.1, 0.5, 0.4] and the actual outputs that take the form [0, 0, 1]. In this case, the common objective function is cross entropy and its variation.

You can modify the objective function to enforce your model to learn a set of parameters with certain properties. As discussed in the section Class Imbalance in Chapter 4, you can modify the objective function to encourage your model to focus on examples of rare classes or examples that are difficult to learn. You can also add regularizers such as L1 and L2 to your loss function to encourage your model to choose parameters of smaller values.  

Each objective function gives you a set of possible values your parameters can take. This set of possible values for parameters is known as the loss surface of a given objective function. A small change to your objective function can give you a very different loss surface, which, in turn, gives you a very different function for your model.

Understanding the possible parameters given by different objective functions can help you choose the objective function that is best suited for your needs. However, this understanding tends to require advanced linear algebra, so it’s common for ML engineers to use popular objective functions that are known to give decent performance for their problems without giving them much thought.

While developing your model, if time permits, you should experiment with different objective functions to see how your model’s behaviors change, both globally on all your data or with respect to different slices of your data. You might be surprised.


### Learning Procedure

Learning procedures the procedures that help your model find the set of parameters that minimize a given objective function for a given set of data, are diverse[^3]. In some cases, the parameters might be calculated exactly. For example, in the case of linear functions, the values of w and b can be calculated from the averages and variances of x and y. In most cases, however, the values of parameters can’t be calculated exactly and have to be approximated, usually via an iterative procedure. For example, K-means clustering uses an iterative procedure called expectation–maximization algorithm. 

The most popular family of iterative procedures today is undoubtedly **gradient descent**. The loss of a model at a given train step is given by the objective function. The gradient of an objective function with respect to a parameter tells us how much that parameter contributes to the loss. In other words, the gradient is the direction that lowers the loss from a current value the most. The idea is to subject that gradient value from that parameter, hoping that this would make the parameter contribute less to the loss, and eventually drive the loss down to 0.

Subtracting the raw gradient values from parameters doesn’t work extremely well. Transforming the gradient values first (such as multiplying the gradient value with 0.003) then subtracting that transformed values from parameters helps models converge much faster. The function that determines how to update a parameter given a gradient value is called an update algorithm, or an **optimizer**. Common optimizers include Momentum, Adam, and RMSProp.

Good optimizers can both speed up your model training process and help your model converge to a better set of parameters. Even though optimizers help your model find the set of parameters that minimize a given objective function for a given set of data, the set of parameters that minimize the loss for your training data isn’t always the best optimizer for you, as you might want the parameters that will perform well on the data your model will encounter in production too[^4]. While developing ML models, especially with gradient descent-based models, it’s often helpful to explore with different types of optimizers. In section AutoML, we’ll discuss how to use ML to find the best optimizers for your model.


<!-- Footnotes themselves at the bottom. -->
## Notes

[^1]:
     In the section AutoML in Chapter 6, we cover how to use algorithms to automatically choose a function form from a predefined set of possible forms.

[^2]:
     If you don’t know what these terms mean, you should still be able to understand approximately 80% of this book. However, I recommend that you take an introductory course to Machine Learning or read an introductory book on Machine Learning in your free time.

[^3]:
     The subfield that studies different learning procedures is called optimization and it’s a large, complex, and fascinating field. Readers interested in learning more can refer to the book [Algorithms for Optimization](https://algorithmsbook.com/optimization/) (Kochenderfer and Wheeler, 2019). 

[^4]:
     In technical terms, you want optimizers that can generalize to unseen data.
