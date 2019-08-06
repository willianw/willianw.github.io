---
title:  "Risk and uncertainty for deep learning"
subtitle: "Errors"
author: "Wferr"
avatar: "img/authors/wferr.png"
image: "img/f.jpg"
date:   2015-04-25 12:12:12
tags: ["machine learning", "statistics"]
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

In a ML problem, we want to aproximate $\hat{f}(X) = Y$. Given that a neural network has weights $w$, we want to maximize the probability $p(Y|w, X)$. During trainning, we adjust $w$ so that $p$ increases. Now for uncertainty we need the posterior probability of weights, i.e., $p(w|Y, X)$. Using Bayes's Theorem:
$$p(w|Y, X) = \frac{p(Y|w, X) \cdot p(w|Y)}{p(X|Y)} = $$
$$= \frac{p(Y|w, X) \cdot p(w|Y)}{p(X|Y)}$$

### The Kullback-Leibler divergence

Given two distributions, $p$ and $q$, we can establish the following similarity [quasimeasure](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):

$$
\begin{align}
KL(q || p) = \sum_{x}&q(x)\log\left[\frac{q(x)}{p(x)}\right], \qquad\text{Discrete case}\\
KL(q || p) = \int_{-\infty}^\infty &q(x)\log\left[\frac{q(x)}{p(x)}\right]dx, \qquad\text{Continuous case}\\
\end{align}
$$

It's important to tell that it's not completely a distance measure, since $KL(q || p) \neq KL(p || q)$.

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

    W0803 13:40:30.582259 140027329709824 deprecation.py:506] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.


    Epoch 1/100
    10000/10000 [==============================] - 2s 229us/step - loss: 0.6072
    Epoch 2/100
    10000/10000 [==============================] - 2s 183us/step - loss: 0.5888
    Epoch 3/100
    10000/10000 [==============================] - 2s 177us/step - loss: 0.5644
    Epoch 4/100
    10000/10000 [==============================] - 2s 175us/step - loss: 0.5402
    Epoch 5/100
    10000/10000 [==============================] - 2s 170us/step - loss: 0.5216
    Epoch 6/100
    10000/10000 [==============================] - 2s 191us/step - loss: 0.4889
    Epoch 7/100
    10000/10000 [==============================] - 2s 182us/step - loss: 0.4685
    Epoch 8/100
    10000/10000 [==============================] - 2s 184us/step - loss: 0.4488
    Epoch 9/100
    10000/10000 [==============================] - 2s 179us/step - loss: 0.4239
    Epoch 10/100
    10000/10000 [==============================] - 2s 177us/step - loss: 0.3834
    Epoch 11/100
    10000/10000 [==============================] - 2s 205us/step - loss: 0.3528
    Epoch 12/100
    10000/10000 [==============================] - 2s 195us/step - loss: 0.3306
    Epoch 13/100
    10000/10000 [==============================] - 2s 212us/step - loss: 0.3331
    Epoch 14/100
    10000/10000 [==============================] - 2s 184us/step - loss: 0.3132
    Epoch 15/100
    10000/10000 [==============================] - 2s 190us/step - loss: 0.3088
    Epoch 16/100
    10000/10000 [==============================] - 2s 187us/step - loss: 0.2999
    Epoch 17/100
    10000/10000 [==============================] - 2s 201us/step - loss: 0.2907
    Epoch 18/100
    10000/10000 [==============================] - 2s 208us/step - loss: 0.2948
    Epoch 19/100
    10000/10000 [==============================] - 2s 202us/step - loss: 0.2851
    Epoch 20/100
    10000/10000 [==============================] - 2s 195us/step - loss: 0.2841
    Epoch 21/100
    10000/10000 [==============================] - 2s 228us/step - loss: 0.2749
    Epoch 22/100
    10000/10000 [==============================] - 2s 201us/step - loss: 0.2741
    Epoch 23/100
    10000/10000 [==============================] - 2s 191us/step - loss: 0.2728
    Epoch 24/100
    10000/10000 [==============================] - 2s 194us/step - loss: 0.2671
    Epoch 25/100
    10000/10000 [==============================] - 2s 187us/step - loss: 0.2651
    Epoch 26/100
    10000/10000 [==============================] - 2s 181us/step - loss: 0.2600
    Epoch 27/100
    10000/10000 [==============================] - 2s 185us/step - loss: 0.2475
    Epoch 28/100
    10000/10000 [==============================] - 2s 184us/step - loss: 0.2450
    Epoch 29/100
    10000/10000 [==============================] - 2s 191us/step - loss: 0.2423
    Epoch 30/100
    10000/10000 [==============================] - 2s 221us/step - loss: 0.2434
    Epoch 31/100
    10000/10000 [==============================] - 2s 184us/step - loss: 0.2261
    Epoch 32/100
    10000/10000 [==============================] - 2s 184us/step - loss: 0.2256
    Epoch 33/100
    10000/10000 [==============================] - 2s 208us/step - loss: 0.2211
    Epoch 34/100
    10000/10000 [==============================] - 2s 174us/step - loss: 0.2146
    Epoch 35/100
    10000/10000 [==============================] - 2s 180us/step - loss: 0.2143
    Epoch 36/100
    10000/10000 [==============================] - 2s 176us/step - loss: 0.2125
    Epoch 37/100
    10000/10000 [==============================] - 2s 181us/step - loss: 0.2018
    Epoch 38/100
    10000/10000 [==============================] - 2s 176us/step - loss: 0.1981
    Epoch 39/100
    10000/10000 [==============================] - 2s 188us/step - loss: 0.2048
    Epoch 40/100
    10000/10000 [==============================] - 2s 183us/step - loss: 0.1953
    Epoch 41/100
    10000/10000 [==============================] - 2s 182us/step - loss: 0.1873
    Epoch 42/100
    10000/10000 [==============================] - 2s 182us/step - loss: 0.1948
    Epoch 43/100
    10000/10000 [==============================] - 2s 192us/step - loss: 0.1795
    Epoch 44/100
    10000/10000 [==============================] - 2s 199us/step - loss: 0.1753
    Epoch 45/100
    10000/10000 [==============================] - 2s 213us/step - loss: 0.1718
    Epoch 46/100
    10000/10000 [==============================] - 2s 203us/step - loss: 0.1812
    Epoch 47/100
    10000/10000 [==============================] - 2s 231us/step - loss: 0.1691
    Epoch 48/100
    10000/10000 [==============================] - 2s 219us/step - loss: 0.1731
    Epoch 49/100
    10000/10000 [==============================] - 2s 202us/step - loss: 0.1696
    Epoch 50/100
    10000/10000 [==============================] - 2s 221us/step - loss: 0.1674
    Epoch 51/100
    10000/10000 [==============================] - 2s 179us/step - loss: 0.1807
    Epoch 52/100
    10000/10000 [==============================] - 2s 205us/step - loss: 0.1607
    Epoch 53/100
    10000/10000 [==============================] - 2s 212us/step - loss: 0.1700
    Epoch 54/100
    10000/10000 [==============================] - 2s 211us/step - loss: 0.1653
    Epoch 55/100
    10000/10000 [==============================] - 2s 216us/step - loss: 0.1647
    Epoch 56/100
    10000/10000 [==============================] - 2s 186us/step - loss: 0.1614
    Epoch 57/100
    10000/10000 [==============================] - 2s 203us/step - loss: 0.1566
    Epoch 58/100
    10000/10000 [==============================] - 2s 245us/step - loss: 0.1592
    Epoch 59/100
    10000/10000 [==============================] - 2s 240us/step - loss: 0.1616
    Epoch 60/100
    10000/10000 [==============================] - 2s 221us/step - loss: 0.1624
    Epoch 61/100
    10000/10000 [==============================] - 2s 204us/step - loss: 0.1602
    Epoch 62/100
    10000/10000 [==============================] - 2s 213us/step - loss: 0.1513
    Epoch 63/100
    10000/10000 [==============================] - 2s 213us/step - loss: 0.1569
    Epoch 64/100
    10000/10000 [==============================] - 2s 195us/step - loss: 0.1499
    Epoch 65/100
    10000/10000 [==============================] - 2s 187us/step - loss: 0.1505
    Epoch 66/100
    10000/10000 [==============================] - 2s 185us/step - loss: 0.1568
    Epoch 67/100
    10000/10000 [==============================] - 2s 190us/step - loss: 0.1481
    Epoch 68/100
    10000/10000 [==============================] - 2s 211us/step - loss: 0.1481
    Epoch 69/100
    10000/10000 [==============================] - 2s 197us/step - loss: 0.1629
    Epoch 70/100
    10000/10000 [==============================] - 2s 208us/step - loss: 0.1449
    Epoch 71/100
    10000/10000 [==============================] - 2s 190us/step - loss: 0.1484
    Epoch 72/100
    10000/10000 [==============================] - 2s 190us/step - loss: 0.1524
    Epoch 73/100
    10000/10000 [==============================] - 2s 207us/step - loss: 0.1477
    Epoch 74/100
    10000/10000 [==============================] - 2s 211us/step - loss: 0.1457
    Epoch 75/100
    10000/10000 [==============================] - 2s 208us/step - loss: 0.1614
    Epoch 76/100
    10000/10000 [==============================] - 2s 242us/step - loss: 0.1489
    Epoch 77/100
    10000/10000 [==============================] - 3s 261us/step - loss: 0.1518
    Epoch 78/100
    10000/10000 [==============================] - 2s 232us/step - loss: 0.1523
    Epoch 79/100
    10000/10000 [==============================] - 2s 243us/step - loss: 0.1557
    Epoch 80/100
    10000/10000 [==============================] - 2s 215us/step - loss: 0.1462
    Epoch 81/100
    10000/10000 [==============================] - 2s 239us/step - loss: 0.1536
    Epoch 82/100
    10000/10000 [==============================] - 2s 228us/step - loss: 0.1539
    Epoch 83/100
    10000/10000 [==============================] - 2s 235us/step - loss: 0.1481
    Epoch 84/100
    10000/10000 [==============================] - 2s 208us/step - loss: 0.1563
    Epoch 85/100
    10000/10000 [==============================] - 2s 214us/step - loss: 0.1511
    Epoch 86/100
    10000/10000 [==============================] - 3s 253us/step - loss: 0.1516
    Epoch 87/100
    10000/10000 [==============================] - 2s 238us/step - loss: 0.1467
    Epoch 88/100
    10000/10000 [==============================] - 2s 202us/step - loss: 0.1522
    Epoch 89/100
    10000/10000 [==============================] - 2s 227us/step - loss: 0.1456
    Epoch 90/100
    10000/10000 [==============================] - 2s 236us/step - loss: 0.1406
    Epoch 91/100
    10000/10000 [==============================] - 2s 206us/step - loss: 0.1493
    Epoch 92/100
    10000/10000 [==============================] - 2s 220us/step - loss: 0.1450
    Epoch 93/100
    10000/10000 [==============================] - 2s 188us/step - loss: 0.1400
    Epoch 94/100
    10000/10000 [==============================] - 2s 191us/step - loss: 0.1451
    Epoch 95/100
    10000/10000 [==============================] - 2s 183us/step - loss: 0.1440
    Epoch 96/100
    10000/10000 [==============================] - 2s 184us/step - loss: 0.1484
    Epoch 97/100
    10000/10000 [==============================] - 2s 188us/step - loss: 0.1471
    Epoch 98/100
    10000/10000 [==============================] - 2s 184us/step - loss: 0.1450
    Epoch 99/100
    10000/10000 [==============================] - 2s 179us/step - loss: 0.1480
    Epoch 100/100
    10000/10000 [==============================] - 2s 187us/step - loss: 0.1441



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
