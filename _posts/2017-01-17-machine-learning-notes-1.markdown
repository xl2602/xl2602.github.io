---
layout: post
title: Machine Learning Notes 1 - Regression
date: 2017-01-17 00:00:00 +0300
color: rgba(255,82,82,0.87) #red
tags: [machine learning, regression] # add tag
categories: machine-learning
---

These are course notes for COMS4721 machine learning course taught by Professor John Paisley ay Columbia University.


# Notation
Each $$ x_i $$ is represented as a vector and attach a 1 to the first dimension of each vector to be convenient

$$
x_i = \begin{bmatrix} 1\\ x_{i1} \\ x_{i2} \\ \vdots\\ x_{id} \\ \end{bmatrix},\qquad
X = \begin{bmatrix} 1 - x_1^T - \\ 1 - x_2^T - \\ \vdots\\ 1 - x_n^T - \\ \end{bmatrix},\qquad
w = \begin{bmatrix} w_0 \\ w_1 \\ \vdots\\ w_d \\ \end{bmatrix},\qquad
y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots\\ y_n \\ \end{bmatrix}
$$



# Linear Regression Model
Generally, the Linear Regression Model can be performed as

$$
y_i\approx f(x_i;w)=\sum_{s=1}^S\mathcal{g}_s(x_i)w_s
$$

- $$ s $$ represents dimension.
- For example, $$ \mathcal{g}_s(x_i) $$ could be $$ x_{ij}^2 $$, or $$ logx_{ij} $$
- A regression method is called linear if the prediction $$f(x_i;w)$$ is a linear function of the unknown parameters $$w$$.

A simple case:

$$
y_i\approx f(x_i;w)=w_0+\sum_{j=0}^dx_{ij}w_j
$$

Prediction

$$
y_{new} \approx x_{new}^Tw_{LS}
$$

### MTH ORDER
![mth-order]({{site.baseurl}}/assets/img/ml-0117-mth-order.png)

# Least Squares

### Objective

$$
\mathcal{L}= \sum_{i=1}^n(y_i-f(x_i;w))^2 = \| y-Xw \|
$$

### Solution

$$
w_{LS} = arg \min_{w} \mathcal{L} = (X^TX)^{-1}X^Ty
$$

- Finds the $$w$$ that minimizes the sum of squared errors
- Assumes $$(X^TX)^{-1}$$ exists, or say $$X^TX$$ has to be a full rank matrix


### Probabilistic View
#### Gaussian
> $$ p(x|\mu, \sigma^2) := \frac{1}{\sqrt{(2\pi)^d\lvert\Sigma\rvert}}\text{exp}\big(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\big)\\
\text{where } \lvert\Sigma\rvert\equiv det(\Sigma) \text{ is the determinant of }\Sigma. $$


#### Model Assumption
$$y \sim N(\mu, \Sigma)$$ with a diagonal covariance matrix $$\Sigma = \sigma^2I$$ and mean $$ \mu = Xw $$

$$ \begin{align}
p(y\mid\mu, \sigma^2)
    &= \frac{1}{\sqrt{(2\pi)^d\mid\sigma^2I\mid}}\text{exp}\big(-\frac{1}{2}(y-Xw)^T(\sigma^2I)^{-1}(y-Xw)\big)\\
    &= \frac{1}{\sqrt{(2\pi)^d\sigma^2}}\text{exp}\big(-\frac{1}{2\sigma}(y-Xw)^T(y-Xw)\big)
\end{align} $$

#### Maximum Likelihood Solution
Ignoring the constants, we can see $$w_{LS}$$ and $$w_{ML}$$ shares the same solution.

$$ \begin{align}
w_{ML}
 &= arg\max_w\ln p(y|\mu=Xw, \sigma^2)\\
 &= arg\max_w-\frac{1}{2\sigma^2}\|y-Xw\|^2-\frac{n}{2}\ln (2\pi\sigma^2)
\end{align} $$

- $$ \mathbb{E}\left[w_{ML} \right] = w $$. $$w_{ML}$$ is an unbiased estimate of $$w$$.
- $$ Var \left[w_{ML} \right] = \sigma^2(X^TX)^{-1} $$. The values of $$w_{ML}$$ are very sensitive to the measured data $$y$$


# Ridge Regression



