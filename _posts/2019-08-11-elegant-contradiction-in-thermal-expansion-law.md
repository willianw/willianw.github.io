---
title:  "Elegant Contradiction in Thermal Expansion Law"
subtitle: "Sometimes a model is just a prediction tool; you can't extract any underlying truth from it"
category: "Physics"
image: "cover.jpeg"
tags: ["Deep"]
---

Do you remember thermal expansion? You might say that in a hotter day, a train rail will be longer than in a colder day.

This is generally written as

$$
\ell = \ell_0 \left( 1 + \alpha\cdot\Delta T \right)
$$

Or more briefly $\frac{\Delta \ell}{\ell_0} = \alpha\ \Delta T$. It describes linear expansion really well. If you search for $\alpha$ values, and collect rail size and temperature variation, you'll be able to predict with great accuracy the total expansion.

But let's come back to maths. Assuming the above formula is right, we might then extrapolate its limits, from $T_0$ to $T_1$:

$$
\dfrac{d\ell}{\ell} = \alpha \cdot dT
\Rightarrow \int_{\ell_0}^{\ell_1} \dfrac{d\ell}{\ell} = \alpha \int_{T_0}^{T_1} dT \\
\ln\left(\dfrac{\ell_1}{\ell_0}\right) = \alpha\left(T_1-T_0\right)
\Rightarrow \ell = \ell_0 \cdot e^{\alpha\Delta T}
$$

### Here things starts becoming strange...
Curious, huh? That doesn't seem to be like that first equation...
That is because we've made a different process. By integration, we change a thermal expansion $T_0 \rightarrow T_1$ for a sequence of several (a.k.a infinite) tiny thermal expansions $T_0,\ T_0 + \delta T,\ T_0+2\delta T,\ \overbrace{\dots}^\infty, T_1-\delta T,\ T_1$. Since the function $\ell$ depends on the path of its arguments, we say that this function is **not conservative**.
Even so, that's very strange for a physical quantity as width. This means, for example, that if you heat a bar from $T_0$ to $T_1$ at once, then cool it first from $T_1$ to $T_{0.5}$, and then from $T_{0.5}$ to $T_0$, the bar will not come back to the same width.

### Mathematically
Let's define $\mathcal{L}(\ell_0, \alpha, \Delta T, n)$ the new length of a bar heaten in $n$ steps.

At the end of 1st step, $\ell_1 = \ell_0 \left( 1 + \alpha\cdot\frac{\Delta T}{n} \right)$

At the end of 2nd step, $\ell_2 = \ell_1 \left( 1 + \alpha\cdot\frac{\Delta T}{n} \right) = \ell_0 \left( 1 + \alpha\cdot\frac{\Delta T}{n} \right)^2$

$\cdots$

At the end of last step, $\mathcal{L} = \ell_0 \left( 1 + \alpha\cdot\frac{\Delta T}{n} \right)^n$.
This is what the above expression looks like:

![png](graph1.png)

**Homework**: do you remember from calculus lessons what happens to $\mathcal{L}$ as $n\rightarrow\infty$? *ANS:* becomes exponential, i.e., $\ell_0 \cdot e^{\alpha\Delta T}$. Surprised? Because I'm not...

### Explanation
The problem is with $\alpha$. Saying $\alpha$ is constant means that matter is eternally extensible. If you've played with a rubber band you know there's a limit for elasticity. The following graph is famous in materials sciences and describes this phenomena:

![png](graph2.jpg)

The plastic region threshold has to do with the average distance between molecules. If you start to put molecules too much away, their coesion forces weaken, and matter becomes irreversibly deformed. If average distance still increases, fracture happens.

This way $\alpha$ should depend on that average molecular distance. So why we don't consider this? Because is too **expensive**. Generally that first formula achieve good results for its pourposes, and its pourposes are related to small temperatures, e.g., what is the width variance of a 700m metal bar given that, in its location, temperature is usually between 11°C and 37°C? For example, if you want to calculate heat expansion for high temperatures, you must consider before this that things melt. 

# Conclusion
Generally models don't have a real link with the truth. They are created just as a manner to calculate quantities of interest. I've chosen an example in which this contradiction is very clear. However, all models have its ontological contradictions with reality.

### Credits
- Photo by Antoine Beauvillain on Unsplash