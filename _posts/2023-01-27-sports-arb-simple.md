---
layout: article
title: "Sports-Betting Arbitrage: The Simple Case"
tags: Python Sports-Betting
permalink: /sports-arb-simple
mathjax: true
mathjax_autoNumber: true
---

Sports betting has become commonplace as of late. What is not commonplace, however, is an understanding of how sports betting works at a fundamental level. This article seeks to illuminate the idea of arbitrage in sports markets, and along the way discuss concepts that are useful even to the casual sports-bettor. After discussing the mathematics behind it all, we implement our ideas in Python so that we can easily find and assess arbitrage opportunities.

<!--more-->

### Overview

#### What are we looking to solve?

We are looking to find and implement a formula that we can use to answer three questions:\\
&emsp; __1)__ How do I know whether a profitable arbitrage is available?\\
&emsp; __2)__ What bets to place in order to take advantage of it?\\
&emsp; __3)__ How profitable will it be?

To begin, let's assume that we have gathered the best odds available for any given outcome -- ideally across many books -- and we organize them into a table such as the following:

$$

\begin{array}{c c} 
& \begin{array}{c c c} \text{Bet }1 & \text{Bet }2 & \text{Bet }3 \\ \end{array} \\
\begin{array}{c c c} \text{Team 1 wins} \\ \text{Tie} \\ \text{Team 2 wins} \end{array} &
\left[
\begin{array}{c c c}
-120 &  &  \\
 & +190 &  \\
 &  & +430
\end{array}
\right]
\end{array} \notag

$$

Clearly, there is only one "best odds" for each outcome, so we can always arrange our data into a format such as the table above. From here, we wish to determine whether there is some combination of bets that will guarantee us a positive profit.

__Note:__ It must be said that we assume that the available bets cover all possible outcomes. We cannot construct an arbitrage if there are outcomes that we cannot bet on.
{:.info}

#### Converting Odds to Payouts

The first step is to convert the above odds into a more interpretable and mathematically-friendly form: payouts.

A bet's payout is simply the amount of money you will have if you bet <span>$</span>1 and you win. The formula is given by:

$$
\text{Payout(Odds)}= \begin{cases}
                    \frac{|\text{Odds}| + 100}{|\text{Odds}|}, & \text{Odds are -} \\
                    \frac{|\text{Odds}| + 100}{100}, & \text{Odds are +}
                     \end{cases}.
                     \notag
$$

Thus, for example, if a bet is $-200$ then payout is $1.5$ which corresponds to a $50\%$ return if the bet is a success. With this, we can more easily use math to help us find the optimal formula.

### The Arbitrage Formula

In this section, we will derive a general formula for an arbitrage portfolio that pays equally across all outcomes using basic linear algebra. The math helps us derive the answer in this section, but in the next section we will connect it more intuitively to financial concepts.

#### Notation

Let $A \in \mathbb{R}^{n \times n}$ be the matrix containing the payouts for each of the $n$ outcomes. Just like above, the columns of $A$ represent the books, and the rows represent each outcome. Similarly, we restrict ourselves to the case where a single bet corresponds to a single outcome -- meaning that we can't have a bet that wins in multiple outcomes -- and we only ever want to look at the most favorable odds available for each outcome. Thus we know that $A$ will be square (there is only 1 best payout per outcome), and furthermore $A$ takes the form

$$\begin{align*}

A = \text{diag}(\vec a),

\end{align*}$$

where $\vec a \in \mathbb{R}^n$, and $a_i>1$ are the payouts of the $i$th outcome. The $0$'s in the non-diagonal entries of $A$ represent the fact that if the given bet doesn't succeed then it pays out $0$. We let $\vec{w} \in \mathbb{R}^n$ denote the weighting of our "portfolio" of bets, and we require that this sums to one.

#### Deriving The Formula

In order to construct an arbitrage that has equal payouts over every outcome, we simply want to find $\vec w$ and some constant scalar $k$ such that the following is satisfied:

$$\begin{align}

A\vec w = \vec 1 k,\\
\vec 1^\intercal \vec w = 1,

\end{align}$$

Clearly, because a ${$1}$ portfolio will turn into ${$k}$ in every outcome, then if $k>1$ then we have a profitable arbitrage. Because $A$ is always invertible, we solve $(1)$ as

$$\vec w = k A^{-1}\vec 1 \notag$$

and we can plug this into $(2)$ to find

$$ k = (\vec 1^\intercal A^{-1} \vec 1)^{-1}.$$

To finish, we then have that

$$\vec w = \frac{A^{-1}\vec 1}{\vec 1^\intercal A^{-1} \vec 1}$$

gives us our arbitrage betting formula.

### A More Intuitive Look

#### Implied (Risk-Neutral) Probability

It isn't necessarily easy to garner intuition of the formula for $\vec w$ at first glance, but by shifting our perspective we can shed some light. First, we must discuss implied (more formally referred to as "risk-neutral") probabilities. Put simply, it is the probability that the betting market has baked into the current odds (or that your sportsbook has set). If an outcome has probability $p$ of occurring, and it pays out $a$, then by placing a ${$1}$ bet on that outcome we have an expected profit of

$$E[\text{Profit}] = p \cdot (a-1) + (1-p) \cdot (-1) \notag$$

If this is positive, then people will keep betting on this side until that is no longer the case. If its negative, people will bet on the other side until that is no longer the case. We find balance only when $E[\text{Profit}]=0$, and thus if the current odds/payout are given by $a$ then the market (on aggregate) views the probability of the outcome to be the value of $p$ such that

$$

E[\text{Profit}] = p \cdot (a-1) + (1-p) \cdot (-1)=0 \notag \\
\implies p=\frac 1a.

$$

#### The Simplified Formula

Using this idea, we can re-write our derivation above in terms of implied probabilities -- replace the $a$'s with $\frac 1p$'s. Then because $A$ is diagonal we have $A=P^{-1}$ where $P \in \mathbb{R}^n$ is a diagonal matrix representing the risk-neutral probabilities (prices) of each bet. With this,

$$\vec w = \frac{P\vec 1}{\vec 1^\intercal P \vec 1},$$

and if we let $\vec p$ denote the diagonal of $P$ just as we let $\vec a$ denote the diagonal of $A$, we can simplify further to get

$$\vec w = \frac{\vec p}{\text{sum}(\vec p)}.$$

In other words, the proper weightings are the implied probabilities scaled so that the weights sum to one! Furthermore, our profit is $k=\frac{1}{\text{sum}(\vec p)}$, meaning that we have a profitable arbitrage if the sum of the implied probabilities is less than one.

#### Why Is This So?

As an example, consider a situation where there are only two bets: $\text{outcome 1}$ has an implied probability of $55\%$ and $\text{outcome 2}$ has an implied probability of $52\%$. There are only two outcomes but their probabilities add up to more than $100\%$! This may be the case because the sportsbooks need to make money and they do so by offering slightly unfavorable bets on all sides. By doing so, the odds are so unfavorable that no profitable arbitrage is possible. You can, however, lock in a sure loss by using the formula for $\vec w$. 

If we consider an asset that pays ${$1}$ if the outcome happens and ${$0}$ if it does not, then it is not difficult to see that the implied probability *is* the market price. So if I pay $55$ cents and $52$ cents for the assets corresponding to $\text{outcomes 1}$ and $2$ respectively, then no matter what happens I paid ${$1.07}$ and won ${$1}$ -- a sure loss.

Now consider if $\text{outcome 1}$ has an implied probability of $55\%$ and $\text{outcome 2}$ has an implied probability of $43\%$. Now, I can pay $55$ cents and $43$ cents respectively, meaning I paid $98$ cents and I always get ${$1}$ -- a sure profit. That is why we need $\text{sum}(\vec p)<1$ for a profitable arbitrage.

__Warning:__ For the active sports bettors out there, this is why its typically not a great idea to make many conflicting bets at the same time -- 99.99% of the time the odds offered by the sportsbook leave absolutely no room for a profitable arbitrage-like situation. Instead, you're far more likely to lock yourself into a losing situation. It is incredibly easy to make a few bets on the same game that contradict each other in subtle ways, and by doing so you are simply decreasing the variance of your outcome. Given how unfavorable the odds usually are, particularly if you make these bets all on the same sportsbook, you are almost certain to lose money.
{:.warning}


#### Calculating the Implied Probability From the Odds

Although this formula follows directly from the discussion above, it is useful to highlight exactly how we can calculate the implied probabilities from the odds. Although some sportsbooks offer this in their app, for those that don't it is incredibly powerful to understand the odds from a probability perspective. The formula is:

$$
\text{Implied Probability(Odds)}= \begin{cases}
                    \frac{|\text{Odds}|}{|\text{Odds}|+100}, & \text{Odds are -} \\
                    \frac{100}{|\text{Odds}| + 100}, & \text{Odds are +}
                     \end{cases}.
                     \notag
$$

__Note:__ If you sportsbet frequently, you should memorize this formula.
{:.info}

### Python Implementation

In this section, we will briefly walk through Python code that can be used to quickly and easily calculate the arbitrage bets, if any exist. First, we will implement some functions to calculate what we discussed above, and then we will create a function that combines everything into one. We will demonstrate with a brief example.

#### The Basic Functions

After we `import numpy as np`, our first step is to implement a function that converts sportsbook odds to implied probabilities as previously discussed. This is done as follows:

```python
def odd_to_price(odd):
    '''
    Converts a sportsbook's odds to implied probability/price.
    '''
    
    if odd >= 0:
        return 100 / (100 + odd)
    else:
        return -odd / (100 - odd)

    
def odds_to_prices(odds):
    '''
    Converts a matrix of sportsbook odds to implied probabilities/prices.
    '''
    return np.vectorize(odd_to_price)(odds)
```

Second, we want to implement our calculations for the arbitrage weights in the form of equation $(6)$. The `notional` parameter refers to the total amount of money you want to bet. Also, if there is no profitable arb available, the function returns nothing.

```python
def calculate_simple_arb(p_vec, notional=100):
    '''
    Given the p-vector holding the implied probalities for each outcome, calculate the arbitrage.
    '''
    
    # Calculate our constant k
    k = 1 / p_vec.sum()
    
    # No need to calculate the weightings if the arb wouldn't be profitable
    if k <= 1:
        print(f"No profitable arb available, return is: {round(100 * (k - 1), 2)}%")
        return
    
    # Print the profitability
    print(f"Arb return: {round(100 * (k - 1), 2)}%")
    
    # Calculate our weights
    w_vec = p_vec * k
    
    # Convert this to a $ amount bet placed
    bets = w_vec * notional
    
    return bets
```

Now, given our odds, we can calculate $\vec p$ and then $\vec w$! 

#### Putting it all together in a more useful way

We can put the above functions together in such a way that makes discovering arbitrage much easier. Consider if we have some bet with $n$ outcomes and we have $b$ different sportsbooks. We can then gather all of the odds data and arrange it into a $n \times b$ matrix such as: 

$$

\begin{array}{c c} 
& \begin{array}{c c c} \text{Book }1 & \text{Book }2 & \text{Book }3 & \text{Book }4\\ \end{array} \\
\begin{array}{c c c} \text{Team 1 wins} \\ \text{Tie} \\ \text{Team 2 wins} \end{array} &
\left[
\begin{array}{c c c}
\; -120 \; & \; -105 \; & \; -125 \; & \; +110 \; \\
+190 & +175 & +230 & +140 \\
+430 & +515 & +300 & +850
\end{array}
\right]
\end{array} \notag

$$

From here, we can automatically:\\
&emsp; __1)__ Find the best prices for each outcome.\\
&emsp; __2)__ Automate the arbitrage calculations.\\
&emsp; __3)__ Return a matrix of the same size that tells us exactly where to bet, and how much.

The code is:

```python
def find_best_prices(prices):
    '''
    Finds the best odds from each book, returns it as an array, and also returns the book indices.
    '''
    
    return prices.min(axis=1), prices.argmin(axis=1)

def simple_arb_from_all_odds(odds, notional=100):
    '''
    Given a rectangular matrix that has all of the odds available for each sportsbook/outcome combination, this function:
    1) finds the best prices/odds for each outcome
    2) calculates the arb weights
    3) returns a corresponding matrix saying exactly how much to bet on each sportsbook/outcome combination 
    '''
    
    # First convert the odds to prices
    prices = odds_to_prices(odds)
    
    # Find the best odds for each outcome, and store which book it came from
    p_vec, books = find_best_prices(prices)
    
    # Calculate the arb bets, if any
    bets = calculate_simple_arb(p_vec, notional)
    
    # If no arb, return None
    if bets is None:
        return
    
    # The matrix specifying exactly where to bet. We first initialize it to be zeros.
    bet_matrix = np.zeros(shape=odds.shape)
    
    # Loop through each outcome, mark which sportsbook/bet combo we need to place our bet on, and how much the bet will be.
    for i, (bet, book) in enumerate(zip(bets, books)):
        
        bet_matrix[i, book] = bet
    
    return bet_matrix
```

Continuing from the example above, if we wanted to bet <span>$</span>100 then the output would be

$$

\begin{array}{c c} 
& \begin{array}{c c c} \text{Book }1 & \text{Book }2 & \text{Book }3 & \text{Book }4\\ \end{array} \\
\begin{array}{c c c} \text{Team 1 wins} \\ \text{Tie} \\ \text{Team 2 wins} \end{array} &
\left[
\begin{array}{c c c}
\;\;\; $0 \;\;\; & \;\;\; $0 \;\;\; & \; $0 \; & $53.84 \\
\; $0 \; & \; $0 \; & $34.26 & \; $0 \; \\
\; $0 \; & \; $0 \; & \; $0 \; & $11.90
\end{array}
\right]
\end{array} \notag

$$

and the return from this arbitrage would amount to $13.06\%$. Thus, we would bet ${$34.26}$ on $\text{Book 3}$ for a $\text{Tie}$, ${$53.84}$ on $\text{Book 4}$ for $\text{Team 1}$ to win, and ${$11.90}$ on $\text{Book 4}$ for $\text{Team 3}$ to win.

__Note:__ Arbitrages of the nature seen in this article are exceedingly rare and typically quite small. For the more complex arbitrage opportunities which are more prevalent, consider reading [this article](/sports-arb-linprog) to learn how to capture them.
{:.info}