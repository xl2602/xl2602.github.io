---
layout: post
title: Machine Learning Notes 1 - Regression
date: 2017-01-17 00:00:00 -0500
color: rgba(255,82,82,0.87) #red
tags: [machine learning, regression] # add tag
categories: machine-learning
---

This is my review notes base on the two machine learning courses I took at Columbia: COMS4721 Machine Learning for Data Science taught by Professor John Paisley and COMSW4995 Applied Machine Learning taught by Professor Andreas Mueller. Most materials are referenced from class slides.


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
![mth-order]({{site.baseurl}}/assets/img/ml/ml-0117-mth-order.png)

# Least Squares

### Objective

$$
\mathcal{L}= \sum_{i=1}^n(y_i-f(x_i;w))^2 = \| y-Xw \|^2
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

- The values in $$w_{ML}$$ may be huge. In general, we often constrain the model parameters $$w$$. According to the regularization scheme, we have Ridge, Lasso etc....

# Ridge Regression - L2

### Objective

$$
\mathcal{L}= \| y-Xw \|^2 + \lambda \| w \|^2
$$

### Solution

$$ \begin{align}
w_{RR} &= arg \min_{w} \mathcal{L} = arg \min_{w} \| y-Xw \|^2 + \lambda \| w \|^2  \\
&= (\lambda I + X^TX)^{-1}X^Ty
\end{align} $$

- $$\lambda$$ is a regularization parameter
- $$ \| w \|^2 $$ is a penalty function that penalizes large values in $$w$$.
- $$\lambda \rightarrow 0: w_{RR} \rightarrow w_{LS} \qquad \lambda \rightarrow \infty: w_{RR} \rightarrow \vec{0}$$.
- Ridge Regression requires data preprocessing: subtract mean and divide by standard deviations

- $$ \mathbb{E}\left[w_{RR} \right] = (\lambda I + X^TX)^{-1}X^TXw $$. $$w_{RR}$$ is an biased estimate of $$w$$.
- $$ Var \left[w_{RR} \right] = \sigma^2Z(X^TX)^{-1}Z^T $$, where $$Z=(I + \lambda (X^TX)^{-1})^{-1}$$ is positive. The variance shrinks.


### Probabilistic View

#### Bayes Rule
> $$ \underbrace{p(w | X, y)}_{posterior} = \frac{\overbrace{p(X, y|w)}^{likelihood} \times \overbrace{p(w)}^{prior}}{\underbrace{p(X|y)}_{evidence}} $$

#### Model Assumption
- The likelihood model: $$y \sim N(Xw, \sigma^2I)$$
- Assume the prior: $$w \sim N(0, \lambda^{-1}I), \quad \text{then } p(w) = \Big( \frac{\lambda}{2\pi} \Big)^{\frac{d}{2}}e^{-\frac{\lambda}{2}w^Tw}$$

#### Maximum A Poseriori (MAP)
Seeks the most probable value $$w$$ under the posterior

$$ \begin{align}
w_{MAP} &= arg \max_{w} \ln p(w|y, X)\\
&= arg \max_{w} \ln \frac{p(y|w, X)p(w)}{p(y|X)}\\
&= arg \max_{w} \ln p(y|w, X) + \ln p(w) - \ln p(y|X)\\
&= (\lambda\sigma^2I+X^TX)^{-1}X^Ty
\end{align} $$

- $$\ln p(y\mid X)$$ does not involve $$w$$, so we can ignore it (when calculating the gradient, it becomes 0.
- $$w_{MAP} = w_{RR}$$ .


### Ridge Regression vs Least Squares
- The Ridge solutions shrinks toward zero $$ \| w_{RR} \| \leq \| w_{LS} \| \text{. Since } w_{RR} = (\lambda I + X^TX)^{-1}X^Ty = \cdots = (\lambda(X^TX)^{-1} + I)^{-1}w_{LS}$$
- Least squares solution: unbiased, but potentially high variance, but Ridge regression solution: biased, but lower variance than LS.
- RR maximizes the posterior, while LS maximizes the likelihood.
- Both ML and MAP are referred to as point estimates of the model parameters that finds a specific value (point) of the vector $$w$$ that maximize the objective function

- LS and RR not suited well for high-dimensional data, since they weight all dimensions without favoring subset of dimensions.


# Least Absolute Shrinkage and Selection Operator (LASSO) - L1

### Objective

$$
\mathcal{L}= \| y-Xw \|^2_2 + \lambda \| w \|_1
$$

### Solution

$$
w_{lasso} = arg \min_{w} \| y-Xw \|^2_2 + \lambda \| w \|_1  \\
\text{where } \| w \|_1 = \sum^d_{j=1} \lvert w_j\rvert
$$

- No local optimal solution
- Natural feature selection
- For correlated features, Lasso is likely to pick one of these at random.


# $$l_p $$ Regression

#### $$ l_p $$-norms

$$
\| w \|_p = \Big( \sum^d_{j=1} \lvert w_j\rvert^p \Big)^{\frac{1}{p}} \qquad \text{for } 0 < p \leq \infty
$$

#### $$ l_p $$-regression

$$
w_{l_p} := arg \min_{w} \| y-Xw \|^2_2 + \lambda \| w \|_p^p  \\
$$

![lp]({{site.baseurl}}/assets/img/ml/ml-0117-lp.png)

#### Solution of $$l_p$$ problem
- $$p<1$$ We can only find an approximate solution (i.e., the best in its “neighborhood”) using iterative algorithms.
- $$p \geq 1, p \neq 2$$ By “convex optimization”. (no “local optimal solutions”, but the true solution can be found exactly using iterative algorithms.
- $$p = 2$$ Ridge.

$$
\begin{array} {lccl}
\text{Method} & \text{Good-o-fit} & \text{penalty} & \text{Solution method}\\
\hline
\text{Least squares} & \| y-Xw \|^2_2 & \text{none} & \text{Analytic solution exists if }X^TX \text{ invertible} \\
\text{Ridge regression} & \| y-Xw \|^2_2 & \| w \|_2^2 & \text{Analytic solution exists always}\\
\text{LASSO} & \| y-Xw \|^2_2 & \| w \|_1 & \text{Numerical optimization to find solution}\\
\end{array}
$$


# Elastic Net

### Objective

$$
\mathcal{L}= \| y-Xw \|^2_2 + \lambda_1 \| w \|_1 + \lambda_2 \| w \|_2^2
$$

* This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge. 
* We control the convex combination of L1 and L2 using the l1_ratio parameter.
* Elastic-net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.


# RANdom SAmple Consensus (RANSAC) 

RANSAC achieves its goal by repeating the following steps<sup>[[1]](http://en.wikipedia.org/wiki/RANSAC)</sup>:
1. Select a random subset of the original data (minimum size to build a model). Call this subset the hypothetical inliers.
2. A model is fitted to the set of hypothetical inliers.
3. All other data are then tested against the fitted model. Those points that fit the estimated model well, according to some model-specific loss function, are considered as part of the consensus set.
4. The estimated model is reasonably good if sufficiently many points have been classified as part of the consensus set.
5. Afterwards, the model may be improved by reestimating it using all members of the consensus set.


RANSAC is
- more robust to outliers
- more common in computer vision tasks

<!-- $$ -->
<!-- \begin{array} {ll} -->
<!-- \text{Probability of choosing an inlier} & P(\text{inlier}) \equiv w \\ -->
<!-- \text{Probability of choosing a sample subset with no outliers} & P(\text{subset with no outlier}) \equiv w ^n \\ -->
<!-- \text{Probability of choosing a sample subset with outliers} & P(\text{subset with outlier(s)}) \equiv 1 - w ^n \\ -->
<!-- \text{Probability of choosing a subset with outliers in all N iterations} & P(\text{N subset with outlier(s)}) \equiv (1 - w ^n)^N \\ -->
<!-- \text{Probability of an unsuccesful run} & P(\text{fail}) \equiv (1 - w ^n)^N \\ -->
<!-- \text{Probability of an succesful run} & P(\text{success}) \equiv 1-(1 - w ^n)^N \\ -->
<!-- \end{array} -->
<!-- $$ -->

<!-- Expected Number of iterations -->
<!-- $$ -->
<!-- N = \frac{\log (1- p(success))}{\log(1-w^n)} -->
<!-- $$ -->
<!-- ###### referenced from http://www.math-info.univ-paris5.fr/~lomn/Cours/CV/SeqVideo/Material/RANSAC-tutorial.pdf -->

<!-- - Typically $$ p$$ set to 0.99 -->
<!-- - we do not know the ratio of outliers in our data set -->


# Huber Loss

$$
\min_{w, \sigma}\sum_{i=1}^n\Big(\sigma+H_m\big(\frac{X_iw-y_i}{\sigma}\big)\sigma\Big)+\alpha\|w\|_2^2 \\


H_m(z) =
\begin{cases}
z^2,  & \text{if }\lvert z \rvert\ < \epsilon, \\
2\epsilon\lvert z \rvert - \epsilon^2, & \text{otherwise}
\end{cases}
$$

![lp]({{site.baseurl}}/assets/img/ml/ml-0117-hl.png){:height="50%" width="50%"}


The Huber Regressor optimizes the squared loss for the samples where $$ \Big\lvert \frac{y - Xw}{\sigma}\Big\rvert < \epsilon $$ and the absolute loss for the samples greater than $$ \epsilon$$ , where $$w$$ and $$ \sigma $$ are parameters to be optimized. The parameter $$ \sigma $$ makes sure that if $$ y $$ is scaled up or down by a certain factor, one does not need to rescale $$ \epsilon $$ to achieve the same robustness. Note that this does not take into account the fact that the different features of $$ X $$ may be of different scales<sup>[[2]](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)</sup>.
- robust to outliers
- less sensitive to small errors, but become linear (sensitive) for large errors.


# (Regularized) Empirical Risk Minimization

$$
\min_{f\in F}\sum_{i=1}^n\underbrace{L\big( f(x_i), y_i \big)}_{\text{Data Fitting}} + \underbrace{\alpha R(f)}_{\text{Regularization}}
$$

To summarize, the machine learning problem can be formalized as minimizing the error on the training set using different loss function (e.g. the squared loss, huber loss), while constraining the model to be simple (control by regularization term).



> ## Resources:
<!-- - Slides of COMS4721 Machine Learning for Data Science taught by Professor John Paisley  at Columbia University. -->
<!-- - Slides of COMSW4995 Applied Machine Learning taught by Professor Andreas Mueller. -->
1. http://en.wikipedia.org/wiki/RANSAC
2. http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html

<!-- - Overview of the RANSAC Algorithm, http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf -->
<!-- - Scikit-learn, http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html -->
<!-- - http://www.math-info.univ-paris5.fr/~lomn/Cours/CV/SeqVideo/Material/RANSAC-tutorial.pdf -->
<!-- - https://en.wikipedia.org/wiki/Random_sample_consensus -->
<!-- - https://en.wikipedia.org/wiki/Huber_loss -->
<!-- - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html -->















