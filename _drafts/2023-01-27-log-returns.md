---
layout: article
title: "Log Returns: What are they and why do we use them?"
tags: FinMath Trading
mathjax: true
mathjax_autoNumber: true
#sidebar:
#    nav: blog-sidebar
---

Log returns are used abundantly in the world of quantitative finance, and they are equally as common in industry as they are in academia. To someone who is encountering them for the first time, log returns may seem to be an odd concept with little use; this article seeks to illustrate how to calculate them, what they are, and why we use them. In short, they provide mathematical and statistical convenience, and have a specific conceptual meaning as well.

<!--more-->

### Notation

We will denote the price of an asset at time $t$ as $P_t$. We denote the return of an asset from time $t_0$ to $t_1$ as $r_{t_0,t_1}$. Typically, we can just set time $t_0=0$ and then represent $r_{0,t}$ in shorthand as $r_t$. However, once we introduce multiple periods it becomes convenient to have both representations.

### Calculating the log return

By now, most of us will have encountered the regular return, or percentage change, formula,

$$\begin{align}

r_t &=\frac{P_t-P_0}{P_0} \notag \\
&=\frac{P_t}{P_0}-1.

\end{align}$$

Note that this article will assume $t$ to be in units of years. The log return, $r^*_t$, is defined as

$$\begin{align}

r^*_t &= \log(1+r_t) \notag \\
&=\log{\frac{P_t}{P_0}}.

\end{align}$$

One can compare the log return and the regular return, and will typically find them to be *very* similar (in fact, one could use Taylor's theorem to bound the difference). So if they are so similar, why do we care to use the log return?

### Continuously compounded return

One benefit of log returns is that they represent the continuously compounded rate of return. More specifically, if we have a regular return $r$ (say over a year, for example), then the log-return $r^\*$ is the equivalent return but expressed as a continously compounded return rate. In other words, a return of $r$ compounded annually is the same as a return rate of $r^*$ compounded continuously, because

$$\begin{align*}

e^{r^*_t}-1 &= e^{\log(1+r_t)}-1 \\
&=r_t.

\end{align*}$$

As such, the log return is simply a different, but equivalent, way of quoting the return. Continuously compounded rates are used frequently in economic and financial theory. They are useful in many ways, such as comparing the rates of two fixed income instruments that have different coupon frequencies.

### Additivity of log returns

A tremendously useful fact about log returns is that the log return over $n$ periods is simply the sum of the log returns for each period. For example, if in year 1 our return is $r_{0,1}$ and in year 2 it is $r_{1,2}$, then our total return $r_{0,2}$ is $(1+r_{0,1})(1+r_{1,2})-1$. Taking the log return,

$$\begin{align*}

r^*_{0,2}&=\log((1+r_{0,1})(1+r_{1,2})-1+1) \\
&=\log(1+r_{0,1})+\log(1+r_{1,2}) \\
&=r^*_{0,1}+r^*_{1,2}

\end{align*}$$

we see that the 2-year log return is the sum of the 1-year log returns. This is property greatly simplifies the mathematics and statistics that we like to use to assess risk and performance. We will see an example of this later.

### Log returns are convenient for statistics

Log returns are a useful concept when one is creating mathematical/statistical models, whether it is for pricing, forecasting, or decomposition. No example is more demonstrative than the Black-Scholes option pricing model, in which a stock's log return is assumed to be normally distributed. It is not that Black and Scholes decided to "use log returns" to build some pricing model -- rather log returns arise naturally when we consider the characteristics a returns distribution must have (or the dynamics that prices follow).

First, it is quite clear that regular returns cannot be normally distributed. A normal random variable can take any value, however asset returns cannot be less than -100%. Log returns, however could theoretically be normally distributed, as $\log(1+r_t) \rightarrow -\infty$ as $r_t \rightarrow -1$.

Now, just because log returns *could theoretically* be normally distributed doesn't necessitate we assume that they are, but the vast majority of statistical models used in any industry assume normality in some form. Why? Because normality is:

    - mathematically simple: it is a well studied distribution with many nice properties.
    - a prevalent phenomenon: normal distributions appear in many places.
    - conceptually practical: things like symmetry often match empirical evidence, and the underlying makeup of what we study tends to justify a normal-like distribution.

The various forms of the central limit theorem tell us that the sum of a large number of random variables tends to have a somewhat normal (if not exactly normal) distribution. Many of the things we model tend to consist of a similar makeup, and so the normality assumption is quite reasonable.

For example, we previously discussed how log returns are the continuously compounded or instantaneous return rate. We also discussed that they are additive. Throwing things like autocorrelation and time-varying volatility out the window, then we might expect annual log returns to be normal, as they are the sum of many random variables. In fact, looking at the SPY ETF, we can assess this hypothesis visually.




I must note that the assumption that regular returns or log returns are normal is extremely dangerous in practice, particularly with regards to risk modeling. Such modeling has been detrimental to many investors/traders in the past. Even if they seem normal it doesn't mean that they are; tail risk is an important consideration.



- convenience of normality. How log returns somewhat match normality while regular returns certainly do not. discuss why, and also why its not perfect. Show historical histogram of yearly and daily/minutely?
- the black-scholes assumptions

#### Log returns for risk and performance assessment

- variance (log returns in practice)
- continous compounded growth
- P_t/P_0 example
- black-scholes log-normality of returns assumptions. (and how this somewhat reflects reality)