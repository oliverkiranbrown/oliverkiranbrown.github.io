---
title: A gentle intro to Statistical Learning Theory
description: This is a post on My Blog about agile frameworks.
tags: ['maths']
date: 2025-05-06
layout: layouts/post.njk
eleventyNavigation:
  key: gentle-intro-to-slt
  parent: writing
---

## Personal Context
*It's been almost a year since my final exams at Warwick. Since then, I've been working at DESNZ and although I've probably learnt some far more useful skills (project management, teamwork in large organisations etc..), I can't deny that I've been missing the maths! I don't want to lose these skills, so as a gentle intro back into the technical weeds, I'll retread some familiar mathematical ground before diving off into some new learning. This post will be an experiment, trying to capture the narrative style of mathematics that I've used to understand different Warwick modules.* 

Behind (most) terse lecture notes lies a beautiful story! Clarissa Poon who taught me SLT in 2023 certainly drew it out in her lectures for *Mathematics of Machine Learning* (forming the basis of this post). However, my goal will be to draw out my own intuition behind the formalism - adding that narrative to the mathematics. 

So, what's the story for SLT?
## Why does Statistical Learning Theory matter?
SLT offers a rigorous mathematical foundation for machine learning. It's a useful framework which can be used to inform model selection decisions. It can help practitioners arrive at a sensible compromise with the bias vs variance trade-off. With the advent of deep learning, this content isn't exactly cutting edge, but spending some time really understanding the ideas is foundational when actually doing ML in the wild. Let's review!

Of the different types of learning (supervised, unsupervised, reenforcement, etc.), SLT is neatest when describing supervised learning. In this post, we'll further simplify to the case of binary classification (photo of a cat or dog) and build up the theory needed to understand ML in this context.
## The set-up
Here, our goal is to make a machine learn from labelled categorical data. More precisely, we want to learn a function $h$ that will take us from our input vector of features $X$ (the words in an email, a customer transaction history, an image etc.) to their corresponding labels (is the email spam or not? is this customer committing fraud or not? is the image a cat or a dog?). If we learn $h:\mathcal{X}\rightarrow \mathcal{Y}$ 'well enough', then we can pass some unlabelled $X$ to $h$ and it'll split out a label $Y$ which we can use to make predictions about the world. 

Small note that our actual data $(X,Y)$ are modelled as random variables, sampled from some probability distribution $\mathcal{X}\times\mathcal{Y}$ with joint measure $P_0$.
## The Very Basics
SLT is all about understanding how well our function $h$ is performing. In order to do this, we have build up some key tools. Please do skip to the interesting stuff if you know this already!
### The Hypothesis Class
Firstly, what is this function $h$? Is the 'model' that everyone talks about. But what does it look like? What shape does it take? We need to draw it from somewhere, to set the guardrails on how the function can look so that the parameters *inside* the function can then be learnt. We call this space of possible functions the hypothesis class, $\mathcal{H}$. 

As a concrete example, $\mathcal{H}$ could look like an indictor function on the interval $[a,b]\subset\mathbb{R}$. An $h\in\mathcal{H}$ would have two parameters, $a$, $b$ on the real line ($a<b$) and would be defined as follows, simply 'indicating' whether the datapoint $x\in\mathbb{R}$ lies inside the interval $[a,b]$ or not. 
$$
h_{a,b}(x)=
\begin{cases} 
1 & \text{if } x\in[a,b] \\
0 & \text{if } x\notin[a,b]
\end{cases}
$$
So here, our hypothesis class $\mathcal{H}$, is the set of all indicator functions on the real line. Formally, 
$$
\mathcal{H}=\{h_{a,b} ; a,b\in\mathbb{R}, a<b\}
$$
In the real world, $\mathcal{H}$ obviously gets a lot more exciting. For example, the hypothesis class for an LLM would be the architecture of a transformer. After defining this model class, you can then do your learning on the parameters *inside* this model class. 
### Loss
We now need some way to quantify how well our chosen function $h\in\mathcal{H}$ is performing. This is how we can understand how to improve our model.

If $h$ is performing well using its current parameters, then it has learnt the phenomena is it modelling - no more work needed! If $h$ is not performing well given its current parameters, then we can use that information to point the parameters of $h$ in a better direction. If we keep doing this (using iterative stochastic gradient descent), the model will keep tweaking its parameters, getting better and better...eventually leading to some pretty stellar ML models. 

To assess performance, we need a *loss function*. This compares the model's predicted label to the real data. If we're correct (or close to correct), then the error (or loss) is low. If our model predicted a label that was *far* from the original value, then our loss in high. 

Formally in our context of binary classification, the loss function $l:\mathcal{Y}\times\mathcal{Y}\rightarrow \mathbb{R}_{}$  takes in two labels:
1) A real label from data, $y$, and
2) A label predicted using our model $\hat{y}$
and outputs a numerical score telling us how good the guess is. 

In our simple context of binary classification, this loss function $l$ could be the simple 'unit loss': if our predicted label is correct, $l$ spits out a 0 - no punishment! If our predicted label is incorrect, then $l$ spits out a 1.  
$$
l(h(x),y)=
\begin{cases}
0 & h(x)=y \\
1 & h(x)\neq y
\end{cases}
$$
### Risk
Now we can quantify the performance of our model $h$, how should we go about choosing it? 

We want our model to perform well on *all* the data, $X$, that could be thrown at it. In other words, we want to minimise the amount of times $h$ is expected to be wrong. In the jargon, we want to minimise the risk $R(h)$. 

Formally, in terms of our random variables $(X,Y)$ drawn from the underlying probability distribution $\mathcal{X}\times\mathcal{Y}$, this is the expectation of the loss function. Literally, 'how often are we expected to be wrong?'
$$
R(h)=\mathbb{E}[l(h(X),Y)]
$$
Digging into the details of what this expectation actually means, for some fixed hypothesis $h$ that we've chosen, we want to travel around our sample space and add up the error our chosen function would give us at each point. By adding up the error at each of the points in our sample space (or integrating), this gives us our expected error or risk:
$$
R(h)=\int_{(x,y)\in(\mathcal{X}\times\mathcal{Y})}l(h(x),y)\text{d}P_0(x,y)=\mathbb{E}[l(h(X),Y)]
$$
### The Bayes Classifier
Now we have the goal: find an $h$ that minimises our risk $R(h)$, the natural next question would be, 'what is the best $h$ we can pick?' 

In our case of simple binary classification (the only possible labels being $\mathcal{Y}=\{0,1\}$), there is an answer! The function that minimises our risk is simply defined to be the *Bayes Classifier*, $h_*$. Formally (assuming this exists and is unique) 
$$
h_*:=\text{argmin}_hR(h)
$$
Importantly, there is *no restriction* on what $h_*$ looks like, or where is is drawn from. The Bayes classifier is simply defined to be the best we can ever possibly do from all possible functions $h$. 
### The Empirical Risk
None of this has been terribly practical so far. We now turn to the actual data which could be collected in the world (spam emails, cat pictures etc). Say we have $n$ data points $(X_i,Y_i)$ which have been drawn identically and independently (iid) from our underlying probability distribution $P_0$. How do we quantify how well our chosen hypothesis $h$ is performing on this real world data?

While before, we integrated over the space of all possible values for $X$, we now have $n$ discrete points, so can just take the average over them! This is the empirical risk $\hat{R}(h)$:
$$
\hat{R}(h)=\frac{1}{n}\sum_{i=1}^n l(h(X_i),Y_i)
$$
From this, again, we ask the question: 'what is the best possible function we can pick to minimise this *empirical risk*'?

While the Bayes classifier was the best possible function we could ever pick to minimise the total loss, we now restrict to only searching over functions *inside our chosen hypothesis class*, $\mathcal{H}$. Now, we are asking the question, 'what is the best function we can possibly choose *inside* our hypothesis class, based on the data we have available?' 

This is the *empirical risk minimiser*, defined to be
$$
\hat{h}\in \text{argmin}_{h\in\mathcal{H}} \hat{R}(h)
$$
This is a far more practical question to answer. We have our chosen hypothesis class $\mathcal{H}$, we have our $n$ data points $\{ X_i,Y_i \}_{i=1}^n$, we just need to pick the best $h$ given these restrictions. 
#### Examples
### Least Square

{% jnbsrcimg "2d-regression", "Test", "Test Plot - working?" %}