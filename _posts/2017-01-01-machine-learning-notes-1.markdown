---
layout: post
title: Machine Learning Notes 1 - Regression
date: 2017-09-12 00:00:00 +0300
color: rgba(255,82,82,0.87) #red
tags: [machine learning, regression] # add tag
categories: machine-learning
---

These are course notes for COMS4721 machine learning course taught by Professor John.




$$ \begin{align}
p(y\mid\mu, \sigma^2)
    &= \frac{1}{\sqrt{(2\pi)^d\mid\sigma^2I\mid}}\text{exp}\big(-\frac{1}{2}(y-Xw)^T(\sigma^2I)^{-1}(y-Xw)\big)\\
    &= \frac{1}{\sqrt{(2\pi)^d\sigma^2}}\text{exp}\big(-\frac{1}{2\sigma}(y-Xw)^T(y-Xw)\big)
\end{align} $$

