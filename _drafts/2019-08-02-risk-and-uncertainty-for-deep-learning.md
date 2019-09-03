---
title:  "Risk and uncertainty for deep learning"
subtitle: "I need to commit last updates made in my other computer"
image: "cover.png"
category: "Machine Learning"
tags: ["notebook"]
---

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('bmh') 
```

# Risk vs. Uncertainty

There are diverse definitions for the distinction between risk and uncertainty.
The most famous comes from Frank Knight, that states:
> Uncertainty must be taken in a sense radically distinct from the familiar notion of Risk, from which it has never been properly separated. The term "risk," as loosely used in everyday speech and in economic discussion, really covers two things which, functionally at least, in their causal relations to the phenomena of economic organization, are categorically different. ... The essential fact is that "risk" means in some cases a quantity susceptible of measurement, while at other times it is something distinctly not of this character; and there are far-reaching and crucial differences in the bearings of the phenomenon depending on which of the two is really present and operating. ... It will appear that a measurable uncertainty, or "risk" proper, as we shall use the term, is so far different from an unmeasurable one that it is not in effect an uncertainty at all. We ... accordingly restrict the term "uncertainty" to cases of the non-quantitive type.

Here we'll use a variant from Ian Osband:
> [...] We identify risk as inherent stochasticity in a model and uncertainty as the confusion over which model parameters apply. For example, a coin may have a fixed $p = 0.5$ of heads and so the outcome of any single flip holds some risk; a learning agent may also be uncertain of $p$.

# The data

We'll use a simulated example for a heteroskedastic random variable $Y$:

$$
y = 2x + 6\sin\left(2\pi x+\frac{\epsilon}{48}\right) \cdot \mathcal{H}(x+2),\\
\epsilon \sim \mathcal{N}\left(0, \ \frac{\pi}{6} + \arctan\left(1.2x + 1\right)\right), \quad x \sim \mathcal{N}(0, 1)
$$

in which $\mathcal{H}$ stands for [Heaviside function](https://en.wikipedia.org/wiki/Heaviside_step_function)


```python
def data_generator(size):
    _x = np.sort(np.random.normal(0, 1, int(size)))
    eps = np.random.normal(0, [-0.5 + 1 / (1 + np.exp(-np.abs(_x))) for x_i in _x], _x.shape)
    _y = _x + np.exp((-_x - 4) * np.pi) + np.abs(_x) * np.sin(np.pi * 2 * (_x + eps / 16)) + eps
    return _x, eps, _y
```


```python
x, eps, y = data_generator(2000)
```


```python
plt.figure(figsize=(20, 5))
plt.scatter(x, y, label='Data')
plt.title('Generated Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show();
```


![png](output_7_0.png)



```python
plt.figure(figsize=(20, 5))
sns.kdeplot(eps, alpha=0.3, shade=True, label='$\epsilon$', c='b')
sns.kdeplot(x, alpha=0.3, shade=True, label='$x$', c='g')
sns.kdeplot(y, alpha=0.3, shade=True, label='$y$', c='r')
plt.title('Distribution of $x$, $\epsilon$, $y$')
plt.xlim(-2, 2)
plt.legend()
plt.show();
```

    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/seaborn/distributions.py:323: MatplotlibDeprecationWarning: Saw kwargs ['c', 'color'] which are all aliases for 'color'.  Kept value from 'color'.  Passing multiple aliases for the same property will raise a TypeError in 3.3.
      ax.plot(x, y, color=color, label=label, **kwargs)



![png](output_8_1.png)



```python
plt.subplots(1, 3, figsize=(20, 3))
plt.subplot(1, 3, 1)
sns.distplot(x)
plt.title('Distribution of $x$')
plt.subplot(1, 3, 2)
sns.distplot(eps)
plt.title('Distribution of $\epsilon$')
plt.subplot(1, 3, 3)
sns.distplot(y)
plt.title('Distribution of $y$')
plt.show();
```


![png](output_9_0.png)


## Simple Regression


```python
def generate_simple_model():
    i = Input((1,))
    x = Dense(1000, activation='relu')(i)
    x = Dense(100, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    o = Dense(1, activation='linear')(x)
    m = Model(i, o)
    return m
```


```python
x, _, y = data_generator(10000)
simple_model = generate_simple_model()
simple_model.compile(loss='mse', optimizer='adam')
hist = simple_model.fit(x, y, epochs=40, verbose=0)
```

    WARNING: Logging before flag parsing goes to stderr.
    W0803 12:08:18.915856 140027329709824 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W0803 12:08:21.801052 140027329709824 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0803 12:08:22.263403 140027329709824 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    W0803 12:08:23.255626 140027329709824 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    W0803 12:08:25.062922 140027329709824 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    W0803 12:08:25.472660 140027329709824 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:973: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    



```python
plt.figure(figsize=(20, 3))
plt.plot(hist.history['loss'])
plt.title('Model trainning progress')
plt.xlabel('# Epochs')
plt.ylabel('loss (MSE)')
plt.ylim(0)
plt.show()
```


```python
_y = simple_model.predict(x)
```


```python
plt.figure(figsize=(20, 5))
plt.scatter(x, y, label='Data', s=1)
plt.plot(x, _y, label='Regression', c='orange')
plt.title('Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show();
```

# Methods

- variational inference
- Monte Carlo Dropout
- BOOTStRAP

## Markov chain Monte Carlo – Metropolis-Hastings


```python

```

## Variational Inference

### Introduction

In a ML problem, we want to aproximate $\hat{f}(X) = Y$. Given that a neural network has weights $w$, we want to maximize the probability $p(Y \mid w, X)$. During trainning, we adjust $w$ so that $p$ increases. Now for uncertainty we need the posterior probability of weights, i.e., $p(w \mid Y, X)$. Using Bayes's Theorem:
$$p(w \mid Y, X) = \frac{p(Y \mid w, X) \cdot p(w \mid Y)}{p(X \mid Y)} = $$
$$= \frac{p(Y \mid w, X) \cdot p(w \mid Y)}{p(X \mid Y)}$$

### The Kullback-Leibler divergence

Given two distributions, $p$ and $q$, we can establish the following similarity [quasimeasure](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):

$$
\begin{align}
KL(q  \Vert  p) = \sum_{x}&q(x)\log\left[\frac{q(x)}{p(x)}\right], \qquad\text{Discrete case}\\
KL(q  \Vert  p) = \int_{-\infty}^\infty &q(x)\log\left[\frac{q(x)}{p(x)}\right]dx, \qquad\text{Continuous case}\\
\end{align}
$$

It's important to tell that it's not completely a distance measure, since $KL(q  \Vert  p) \neq KL(p  \Vert  q)$.

## Monte-Carlo Dropout


```python
def generate_dropout_model():
    i = Input((1,))
    x = Dense(1000, activation='relu')(i)
    x = Dropout(0.1)(x, training=True)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.1)(x, training=True)
    x = Dense(10, activation='relu')(x)
    x = Dropout(0.1)(x, training=True)
    o = Dense(1, activation='linear')(x)
    m = Model(i, o)
    return m
```


```python
x, _, y = data_generator(10000)
dropout_model = generate_dropout_model()
dropout_model.compile(loss='mse', optimizer='adam')
hist = dropout_model.fit(x, y, epochs=100, verbose=1)
```

```python
plt.figure(figsize=(20, 3))
plt.plot(hist.history['loss'])
plt.title('Model trainning progress')
plt.xlabel('# Epochs')
plt.ylabel('loss (MSE)')
plt.ylim(0)
plt.show()
```


![png](output_28_0.png)



```python
preds = []
for i in range(10):
    _y = dropout_model.predict(x)
    preds.append(_y)
predictions = np.array(preds)

plt.figure(figsize=(20, 5))
plt.scatter(x, y, label='Data', s=5)
for i, _y in enumerate(predictions):
    plt.scatter(x, _y, label=f'Regression #{i+1}', c='orange', s=1, alpha=0.1)
plt.title('Regression w/ Dropout')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show();
```


![png](output_29_0.png)



```python
mean = predictions.mean(axis=0).flatten()
stdv = predictions.std(axis=0).flatten()

plt.figure(figsize=(20, 5))
plt.plot(x, mean, label='Mean', ms=1, color='orange')
plt.fill_between(x, mean-stdv, mean+stdv, label='Stdev', alpha=0.4)
plt.title('Error Estimation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show();
```


![png](output_30_0.png)


# References

- https://arxiv.org/pdf/1902.10189.pdf
- https://gdmarmerola.github.io/risk-and-uncertainty-deep-learning/  
- https://arxiv.org/pdf/1905.09638.pdf  
- https://arxiv.org/pdf/1505.05424.pdf  
- https://arxiv.org/pdf/1506.02142.pdf  
- https://arxiv.org/pdf/1602.04621.pdf  
- https://arxiv.org/pdf/1806.03335.pdf  
- https://arxiv.org/pdf/1505.05424.pdf  
- https://arxiv.org/pdf/1601.00670.pdf  
- https://ermongroup.github.io/cs228-notes/inference/variational/  
- https://github.com/ericmjl/website/blob/master/content/blog/variational-inference-with-pymc3-a-lightweight-demo/linreg.ipynb  
- https://medium.com/tensorflow/regression-with-probabilistic-layers-in-tensorflow-probability-e46ff5d37baf  
- https://papers.nips.cc/paper/8080-randomized-prior-functions-for-deep-reinforcement-learning.pdf  
- http://proceedings.mlr.press/v37/salimans15.pdf
- https://www.cs.ubc.ca/~schmidtm/Courses/540-W18/L34.pdf
- http://bayesiandeeplearning.org/2016/papers/BDL_4.pdf  
- https://towardsdatascience.com/uncertainty-estimation-for-neural-network-dropout-as-bayesian-approximation-7d30fc7bc1f2  


```python

```
