# Introduction to Neural Networks. Multi-Layered Perceptron

In the previous section, you learned about the simplest neural network model - one-layered perceptron, a linear two-class classification model.

In this section we will extend this model into a more flexible framework, allowing us to:

* perform **multi-class classification** in addition to two-class
* solve **regression problems** in addition to classification
* separate classes that are not linearly separable

We will also develop our own modular framework in Python that will allow us to construct different neural network architectures.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ai/quiz/7)

## Formalization of Machine Learning

Let's start with formalizing the Machine Learning problem. Suppose we have a training dataset **X** with labels **Y**, and we need to build a model *f* that will make most accurate predictions. The quality of predictions is measured by **Loss function** &lagran;. The following loss functions are often used:

* For regression problem, when we need to predict a number, we can use **absolute error** &sum;<sub>i</sub>|f(x<sup>(i)</sup>)-y<sup>(i)</sup>|, or **squared error** &sum;<sub>i</sub>(f(x<sup>(i)</sup>)-y<sup>(i)</sup>)<sup>2</sup>
* For classification, we use **0-1 loss** (which is essentially the same as **accuracy** of the model), or **logistic loss**.

For one-level perceptron, function *f* was defined as a linear function *f(x)=wx+b* (here *w* is the weight matrix, *x* is the vector of input features, and *b* is bias vector). For different neural network architectures, this function can take more complex form.

> In the case of classification, it is often desirable to get probabilities of corresponding classes as network output. To convert arbitrary numbers to probabilities (eg. to normalize the output), we often use **softmax** function &sigma;, and the function *f* becomes *f(x)=&sigma;(wx+b)*

In the definition of *f* above, *w* and *b* are called **parameters** &theta;=⟨*w,b*⟩. Given the dataset ⟨**X**,**Y**⟩, we can compute an overall error on the whole dataset as a function of parameters &theta;.

> ✅ **The goal of neural network training is to minimize the error by varying parameters &theta;**

## Gradient Descent Optimization

There is a well-known method of function optimization called **gradient descent**. The idea is that we can compute a derivative (in multi-dimensional case called **gradient**) of loss function with respect to parameters, and vary parameters in such a way that the error would decrease. This can be formalized as follows:

* Initialize parameters by some random values w<sup>(0)</sup>, b<sup>(0)</sup>
* Repeat the following step many times:
    - w<sup>(i+1)</sup> = w<sup>(i)</sup>-&eta;&part;&lagran;/&part;w
    - b<sup>(i+1)</sup> = b<sup>(i)</sup>-&eta;&part;&lagran;/&part;b

During training, the optimization steps are supposed to be calculated considering the whole dataset (remember that loss is calculated as a sum through all training samples). However, in real life we take small portions of the dataset called **minibatches**, and calculate gradients based on a subset of data. Because subset is taken randomly each time, such method is called **stochastic gradient descent** (SGD).

## Multi-Layered Perceptrons and Backpropagation

One-layer network, as we have seen above, is capable of classifying linearly separable classes. To build a richer model, we can combine several layers of the network. Mathematically it would mean that the function *f* would have a more complex form, and will be computed in several steps:
* z<sub>1</sub>=w<sub>1</sub>x+b<sub>1</sub>
* z<sub>2</sub>=w<sub>2</sub>&alpha;(z<sub>1</sub>)+b<sub>2</sub>
* f = &sigma;(z<sub>2</sub>)

Here, &alpha; is a **non-linear activation function**, &sigma; is a softmax function, and parameters &theta;=<*w<sub>1</sub>,b<sub>1</sub>,w<sub>2</sub>,b<sub>2</sub>*>.

The gradient descent algorithm would remain the same, but it would be more difficult to calculate gradients. Given the chain differentiation rule, we can calculate derivatives as:

* &part;&lagran;/&part;w<sub>2</sub> = (&part;&lagran;/&part;&sigma;)(&part;&sigma;/&part;z<sub>2</sub>)(&part;z<sub>2</sub>/&part;w<sub>2</sub>)
* &part;&lagran;/&part;w<sub>1</sub> = (&part;&lagran;/&part;&sigma;)(&part;&sigma;/&part;z<sub>2</sub>)(&part;z<sub>2</sub>/&part;&alpha;)(&part;&alpha;/&part;z<sub>1</sub>)(&part;z<sub>1</sub>/&part;w<sub>1</sub>)

> ✅ The chain differentiation rule is used to calculate derivatives of the loss function with respect to parameters.

Note that the left-most part of all those expressions is the same, and thus we can effectively calculate derivatives starting from the loss function and going "backwards" through the computational graph. Thus the method of training a multi-layered perceptron is called **backpropagation**, or 'backprop'.

<img alt="compute graph" src="images/ComputeGraphGrad.png"/>

> TODO: image citation

> ✅ We will cover backprop in much more detail in our notebook example.  

## Conclusion

In this lesson, we have built our own neural network library, and we have used it for a simple two-dimensional classification task.

## 🚀 Challenge

In the accompanying notebook, you will implement your own framework for building and training multi-layered perceptrons. You will be able to see in detail how modern neural networks operate.

Proceed to the [OwnFramework](OwnFramework.ipynb) notebook and work through it.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ai/quiz/8)

## Review & Self Study

Backpropagation is a common algorithm used in AI and ML, worth studying [in more detail](https://wikipedia.org/wiki/Backpropagation)

## [Assignment](lab/README.md)

In this lab, you are asked to use the framework you constructed in this lesson to solve MNIST handwritten digit classification.

* [Instructions](lab/README.md)
* [Notebook](lab/MyFW_MNIST.ipynb)

## Applied lab: Single Perceptron Update — Offline Service Adapter (R38-T045)

Before constructing multi-layered perceptrons and backpropagation frameworks, it is essential to understand the elementary building block: a single perceptron prediction and parameter update step.

This applied software-engineering lab illustrates:
- **Weighted Sum:** How inputs and weights form linear combinations $z = \sum w_i x_i + b$.
- **Classification Threshold & Zero-Score Tie Rule:** Predictions use step activation $\hat{y} = 1$ if $z > 0$ else $0$. A score of exactly zero ties in favor of class 0.
- **Perceptron Learning Rule:** Error $e = y - \hat{y}$ drives parameter updates via $w_i \leftarrow w_i + \eta \cdot e \cdot x_i$ and $b \leftarrow b + \eta \cdot e$.
- **Callable Interface & Dependency Injection:** Decouples the update logic from any specific inference provider by passing a callable predictor: `predictor(weights, bias, inputs) -> prediction`.
- **Deterministic Local Adapter:** Evaluates predictions locally and deterministically without network access.
- **Where the Simplification Stops:** This lab evaluates a single deterministic step on a 2D fixture; it does not train a complete multilayer neural network over multiple epochs, nor does it establish network communication, authentication, or remote cloud SLA guarantees.

**Main Fixture:**
- Initial weights: `[0, 0]`, bias: `0`, input: `[1, 2]`, target: `1`, learning rate: `1`
- Calculation: $z = (0 \cdot 1) + (0 \cdot 2) + 0 = 0 \implies \hat{y} = 0, \text{error} = 1 - 0 = 1$
- Updated weights: `[1, 2]`, updated bias: `1`

**Run (from repository root):**

    python3 lessons/3-NeuralNetworks/04-OwnFramework/applied-labs/r38-t045/example.py

**Run automated tests:**

    python3 -m unittest discover -s lessons/3-NeuralNetworks/04-OwnFramework/applied-labs/r38-t045 -p "test_*.py"

See the complete implementation, tests, and documentation:
- Code: [applied-labs/r38-t045/example.py](applied-labs/r38-t045/example.py)
- Tests: [applied-labs/r38-t045/test_example.py](applied-labs/r38-t045/test_example.py)
- Documentation: [applied-labs/r38-t045/README.md](applied-labs/r38-t045/README.md)
