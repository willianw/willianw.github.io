---
title:  "Risk and uncertainty for deep learning"
subtitle: "Ok, you've got your results. But how dispersive will your predictions be?"
image: "cover.png"
category: "Machine Learning"
tags: ["notebook"]
---

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

# Strategies for output distribution
In all the following cases, there are some parameters $\theta$ (model parameters, such as the neural network weights), and some data $D = \{x, y\}$. For output distribution, we need to know $P(\theta\mid D)$, i.e., the probability of having such weights given the data. By Bayes Theorem, we have

![png](bayes.png)

Generally, the last integral is either difficult or expensive to calculate. So the following methods will be able to approximate the posterior distribution.

- Variational Inference
- Monte Carlo Dropout
- BOOTSTRAP

## Markov chain Monte Carlo – Metropolis-Hastings

Markov chain Monte Carlo (MCMC) is a group of methods to generate a sample from a unknown distribution. They combine Monte Carlo techniques for sampling from a generated distribution and Markov chains to calculate the probability of each value in the distribution.
Between MCMC methods, the most famous is Metropolis-Hastings, represented by the following algorithm:

1. Get a base distribution $f$ and a candidate distribution $g$. Generally $f$ is the product $P(D\mid \theta)P(\theta)$, which is well known; as $g$ is a simple distribution (such as gaussian). As we iterate, $g$ will accumulate more samples, and become more similar to $P(\theta\mid D)$
2. For each iteration t:
    1. we have $x_t$ as the previous sample from $g$
    2. get a sample $x$ from $g(x\mid x_t)$
    3. calculate the acceptance ratio $\alpha = f(x)/f(x_t)$. It gives an idea of how probable $x$ is among the previously gathered samples.
    4. with probability $\alpha$, insert $x$ to the gathered samples. Else reinsert the previous sample $x_t$.

This algorithm is proven to asymptotically approximate $P(\theta\mid D)$. However, it is still extremely expensive computationally, being sometimes infeasible. For that reason, I wasn't able to compute this algorithm for a DNN (because it has many parameters). So, I've changed the model to a 3-degree polinomial. I've also used a framework called pymc3, for applying Monte Carlo methods. Obs.: as we use Metropolis-Hastings, the sample function doesn't get just samples, it updates the parameter values (analogous to a fit method).

```python
polynoms_n = range(10)
with pm.Model() as model:
    # Priors
    theta = [pm.Normal(f'theta_{i}', 0, sigma=20) for i in polynoms_n]

    # Likelihood
    likelihood = pm.Normal('y', sum([theta[i] * x ** i for i in polynoms_n]),
                        sigma=1, observed=y)
    # Inference!
    trace = pm.sample(10000, cores=3, step=pm.Metropolis())
```

![png](mcmc_params.png)

![png](mcmc_results.png)

## Variational Inference

### Introduction

In a ML problem, we want to aproximate $\hat{f}(X) = Y$. Given that a neural network has weights $w$, we want to maximize the probability $p(Y \mid w, X)$. During trainning, we adjust $w$ so that $p$ increases. Now for uncertainty we need the posterior probability of weights, i.e., $p(w \mid Y, X)$. Using Bayes's Theorem:
$$p(w \mid Y, X) = \frac{p(Y \mid w, X) \cdot p(w \mid Y)}{p(X \mid Y)} = $$
$$= \frac{p(Y \mid w, X) \cdot p(w \mid Y)}{p(X \mid Y)}$$

### The Kullback-Leibler divergence

Given two distributions, $p$ and $q$, we can establish the following similarity [quasimeasure](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):

![png](kl.png)

It's important to tell that it's not completely a distance measure, since $KL(q  \Vert  p) \neq KL(p  \Vert  q)$.

```python
def variational_network(X_train, y_train):
    # Initialize random weights between each layer
    hidden_shape = (X_train.shape[1], 1000, 100, 10, 1)
    layers_init = [
        np.random.randn(hidden_shape[i], hidden_shape[i+1]).astype(float)
        for i in range(len(hidden_shape)-1)
    ]
    with pm.Model() as neural_network:
        model_input = theano.shared(X_train)
        model_output = theano.shared(y_train)
        layers = [model_input]
        for i, layer in enumerate(layers_init):
            weights = pm.Normal(f'w_{i+1}', 0, sigma=1,
                                shape=layer.shape,
                                testval=layer)
            layers.append(weights)
        act = layers[0]
        for layer in layers[1:]:
            act = pm.math.sigmoid(pm.math.dot(act, layer))

        output = pm.Normal('out',
                           act, sigma=1,
                           observed=model_output,
                           total_size=y_train.shape[0]
                          )
    return model_input, model_output, neural_network

model_input, model_output, neural_network = variational_network(X_train, y_train)
```

![png](vi_results.png)

Pretty bad, huh? That's because

1. the DNN isn't well optimized, it would take more several training steps to get accurate;
2. variational inference is susceptible to get stuck in local optima.

## Monte-Carlo Dropout

Methods #1 #2, although not working well for DL, are well consolidated for little data (standard statistics). In 2016, this paper showed that using dropout in every layer at prediction time is a guaranteed way to determine uncertainty, i.e., is equivalent to a Monte Carlo process.

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

# Application — Predicting Car price Distribution

I've used a [Kaggle dataset](https://www.kaggle.com/avikasliwal/used-cars-price-prediction) that contains a list of used cars, its prices, and features as power, how many times the car was sold, mileage, etc. Here is some preprocessing:

```python
# Fuel
df = pd.get_dummies(df, columns=['Fuel_Type'], prefix='fuel_')

# Transmission
df['Manual'] = (df['Transmission'] == 'Manual').astype(int)

# Owner type (first, second, ...)
df['Owner_Type'] = df['Owner_Type'].map({'First': 1, 'Second': 2, 'Third': 3}).fillna(4)

# Mileage
df['Mileage'] = df['Mileage'].apply(process_mileage)
df['Mileage'] = df['Mileage'].fillna(df['Mileage'].mean())

# Engine
df['Engine'] = df['Engine'].astype('str').str.extract('(\d+)')[0].astype('float')
df['Engine'] = df['Engine'].fillna(df['Engine'].mean())

# Power
df['Power'] = df['Power'].astype('str').str.extract('(\d+)')[0].astype('float')
df['Power'] = df['Power'].fillna(df['Power'].mean())

# Seats
df['Seats'] = df['Seats'].fillna(df['Seats'].median())

# Drop others
df.drop(['Name', 'Location', 'New_Price', 'Transmission'], axis=1, inplace=True)
```

And then we train using the following model:

```python
def generate_dropout_model(input_shape):
    i = Input((input_shape,))
    x = Dense(1024, kernel_initializer='normal', activation='relu')(i)
    x = Dropout(0.1)(x, training=True)
    x = Dense(512, kernel_initializer='normal', activation='relu')(x)
    x = Dropout(0.1)(x, training=True)
    x = Dense(256, kernel_initializer='normal', activation='relu')(x)
    x = Dropout(0.1)(x, training=True)
    x = Dense(128, kernel_initializer='normal', activation='relu')(x)
    x = Dropout(0.1)(x, training=True)
    x = Dense(64, kernel_initializer='normal', activation='relu')(x)
    x = Dropout(0.1)(x, training=True)
    x = Dense(32, kernel_initializer='normal', activation='relu')(x)
    x = Dropout(0.1)(x, training=True)
    o = Dense(1, kernel_initializer='normal', activation='linear')(x)
    m = Model(i, o)
    return m
```

![png](training_hist.png)

## Results

Here are some predicted values distributions. We get a sample from the model with price predictions outputs. Then we check it vs. the real price predicted.

![png](results1.png)

![png](results2.png)

![png](results3.png)

# Conclusion
There are several (some not so good) more approaches for estimating uncertainty not discussed here, some of them useful for deep learning too.
You can check the complete code in the link below!

{% include repo.html name='risk-and-uncertainty-deep-learning'  %}

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
