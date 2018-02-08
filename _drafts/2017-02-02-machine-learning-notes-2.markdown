---
layout: post
title: Machine Learning Notes 2 - Classification
date: 2017-02-02 00:00:00 -0500
color: rgba(255,82,82,0.87) #red
tags: [machine learning, classification] # add tag
categories: machine-learning
---

This is my review notes base on the two machine learning courses I took at Columbia: COMS4721 Machine Learning for Data Science taught by Professor John Paisley and COMSW4995 Applied Machine Learning taught by Professor Andreas Mueller. Most materials are referenced from class slides.


# Classifier
Classifier is the function $$f$$ used in classification task to map input $$x$$ to class $$y$$.

$$
y=f(x): f\text{ takes in }x \in \mathcal{X} \text{ and declares its class to be }y\in \mathcal{Y}\\
$$


# K-Nearest Neighbors

### Algorithms

For a new input x.
1. Return the k points closest to x, indexed as $$ x_{i_1}, \ldots ,x_{i_k} $$.
2. Return the majority-vote of $$ y_{i_1}, \ldots ,y_{i_k} $$. (Break ties both steps arbitrarily)

* The default distance is Euclidean Distance: $$ \|u-v\|_2 = (\sum^d_{i=1}(u_i-v_i)^2)^{\frac{1}{2}} $$, but it could be any other distances.
* Smaller k -> fragment, bigger k -> more smooth and predictions are more "stable"

# Bayes Classifier

TBD


#