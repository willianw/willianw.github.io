---
title:  "The dog and rabbit chase"
subtitle: "Never thought it was so difficult to catch a rabbit"
category: "Physics"
image: "cover.jpg"
tags: ["challenge"]
---

Irodov's problem

From the very famous Irodov's book, here follows the statement:
> A rabbit runs in a straight line with a speed $u$. A dog with a speed $V > u$ starts to pursuit it and during the pursuit always runs in the direction towards the rabbit. Initially the rabbit is at the origin while the dog’s coordinates are $(0, L)$. After what time does the dog catch the rabbit?

$$\dot{\vec{r}_r}(t) = u\left( 0, 1 \right)$$

$$\dot{\vec{r}}_d(t) = V \left( -\cos\theta, \sin\theta \right),\quad \tan\theta = \frac{x_r-x_d}{y_r}$$

That definetively gives us a very complicated diferential equation. We have that $x_d(t) = ut$, $x_r(t) = \int_0^tV\sin\theta\cdot dt$ and $y_r(t) = L - \int_0^tV\cos\theta\cdot dt$. Thus

$$
\tan(\theta) = \dfrac{\int_0^tV\sin\theta\cdot dt - ut}{L - \int_0^tV\cos\theta\cdot dt}
$$

Let $A = \int_0^tV\sin\theta\cdot dt$ and $y_r(t) = L - \int_0^tV\sqrt{1-\sin^2\theta}\cdot dt$

![png]({% include img_path.html a=page.path p="statement.png" %})

# Credits and References
- I. E. Irodov, Problems in General Physics