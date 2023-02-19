---
layout: article
title: "Sports-Betting Arbitrage: Complex Arbitrage Through Linear Programming"
tags: Python Sports-Betting Optimization Numerical-Methods
permalink: sports-arb-linprog
mathjax: true
mathjax_autoNumber: true
---

Basic arbitrage is remarkably difficult to find in sports markets these days. More complex arbitrage strategies, however, are more prevalent. This article builds out the idea of using numerical methods (specifically linear programming) to find more complex arbitrage strategies. At the end, we implement this idea in Python and show an example.

<!--more-->

### The Setup

The foundation of what we refer to as a "complex arbitrage" in this article is that notion certain bets may pay out across multiple outcomes. Different sportsbooks often have different ways that they like to group outcomes, and as a result the bets offered don't always match up. This prevents us from using the simple formula we found in the article on sports-betting [arbitrage for simpler cases](/sports-arb-simple). In fact, there often is no analytical formula for arbitrage -- instead we must use optimization algorithms to find the optimal bets to place. Much of this article will rely on ideas introduced in the first article, so check it out if the ideas below are unclear.

#### Notation

Let $$A = \begin{bmatrix} \vec a_1 \\ \vdots \\ \vec a_n \end{bmatrix} \in \mathbb{R}^{n \times b}$$ be the matrix containing the payouts for each of the $n$ outcomes and the $b$ available bets that can be placed. For example, let $\text{Book 1}$ offer an under $5$ goals scored bet where an outcome of $5$ goals results in a push (money back). Let $\text{Book 2}$ offer both over and under $5.5$ goals scored bets, and let $\text{Book 3}$ offer a bet that pays only if there is exactly $5$ goals scored. Then our payout matrix might look like

$$
A = 
\begin{array}{c c} 
& \begin{array}{c c c} \text{Bet }1 & \text{Bet }2 & \text{Bet }3 & \text{Bet }4 \end{array} \\
\begin{array}{c c c} \text{Under 5 Goals} \\ \text{5 Goals} \\ \text{Over 5 Goals} \end{array} &
\left[
\begin{array}{c c c}
\;\;\; 1.55 \;\;\; & \;\;\; 0 \;\;\; & \;\;\; 1.8 \;\;\; & \;\;\; 0 \;\;\; \\
1 & 0 & 0 & 11 \\
0 & 3 & 0 & 0
\end{array}
\right]
\end{array}. \notag
$$

Also, we let $\vec w \in \mathbb{R}^b$ denote the portfolio weighting of each bet. As such, we require that $\vec w$ sums to one, or $\vec 1 ^\intercal \vec w = 1$. In addition, we cannot "short" any bets, meaning $\vec w \geq 0$. Lastly, we will use the notation $A_{-j} \in \mathbb{R}^{(n-1) \times b}$ to denote the matrix $A$ with its $j$-th row removed (the vector $\vec a_j$ is removed).

__Note:__ This approach assumes that the $n$ outcomes are mutually exclusive and span the entire set of possible outcomes. We cannot have arbitrage if there are outcomes that we cannot bet on.
{:.info}

### The Problem

From here, we can move to solving the problem. Depending on how we set things up, our objective function will differ slightly, however they are all still arbitrages. Below, we walk through a few different ideas of how to set the problems up -- and of course, how to solve them.

#### Solving The Problem

Problems of the forms that are seen below are called linear programs, and luckily their are many optimization algorithms available to solve linear programming problems. In our implementation, we will use SciPy's optimization package in Python which has a built in linear programming solver. In doing so, we will have to group all of our inequality constraints together into one big matrix, and likewise with our equality constraints. Part of the art of solving optimization problems is framing them in such a way that we can actually solve them. 

__Note:__ Sometimes, *there is no solution*. This happens when our constraints are so restrictive that no $\vec w$ is capable of satisfying them all. This means that no arbitrage is possible. Although unlikely, it is also possible that there are multiple solutions, where multiple betting strategies have the exact same result.
{:.info}

#### Maximizing A Specific Outcome ($\text{Problem #1}$)

To have arbitrage, we must have no outcomes that would result in a loss, and at least one or more outcome(s) that would result in a positive profit. There are numerous ways that we could express our desire for profit mathematically, and we will explore a few of these. For now, let's assume that we simply want to maximize our profit in outcome $j$. Our objective is thus as follows:

$$
\begin{align}
\text{Maximize: }& \vec a_j ^ \intercal \vec w \\
\text{such that:}& \notag \\
& \vec 1 ^\intercal \vec w = 1 \notag \\
& \vec w \geq 0 \notag \\
& A \vec w \geq 1 \notag
\end{align}
$$

#### Making Equal-Payouts ($\text{Problem #2}$)

Ideally, we can recieve a profit in *every* outcome. Fortunately, we can use linear programming to solve for these arbitrages as well. The caveat is that these opportunities are much more rare. Now, our objective becomes

$$
\begin{align*}
\text{Maximize: }& \vec a_j ^ \intercal \vec w \\
\text{such that:}& \\
& \vec 1 ^\intercal \vec w = 1 \\
& \vec a_1 ^ \intercal \vec w = \vec a_2 ^ \intercal \vec w = \dots = \vec a_n ^ \intercal \vec w \\
& \vec w \geq 0
\end{align*}
$$

where now our choice of $j$ is trivial. In order to use linear programming, we need to put things in a different form. Note that the last constraint is equivalent to $\vec a_i ^ \intercal \vec w = \vec a_j ^ \intercal \vec w$ for $i=1,2, \dots, n$. Dropping the redundant $j$-th row, and letting $$J = \begin{bmatrix} \vec a_j \\ \vdots \\ \vec a_j \end{bmatrix} \in \mathbb{R}^{(n-1) \times b}$$, then this constraint is written as

$$
A_{-j} \vec w = J \vec w \\
\implies (A_{-j}-J) \vec w = 0.
\notag
$$

With this, our objective is

$$
\begin{align}
\text{Maximize: }& \vec a_j ^ \intercal \vec w \\
\text{such that:}& \notag \\
& \vec 1 ^\intercal \vec w = 1 \notag \\
& (A_{-j}-J) \vec w = 0 \notag \\
& \vec w \geq 0 \notag \\
\end{align}
$$

and we can proceed with our linear programming algorithms. 

__Note:__ The formulation above could technically have a solution leading to a guaranteed loss. Although we could choose to prevent this, sometimes it can be useful to see a loss. Creating the *best* possible even-payout arbitrage (which could be negative) provides a way to assess how large the pricing discrepancies are between books/bets, which is closely related to how fair the pricing is in general.
{:.info}

#### Maximizing Expected Profit ($\text{Problem #3}$)

To add some flavor to the discussion, and to illustrate the breadth of possibilities available, we will demonstrate how to use linear programming to maximize the expected profit earned while still ensuring no losses. Note that in order to utilize this method, one must have a prior distribution for the outcomes, meaning we must assign a probability to each outcome. Realistically, this approach would best serve situations in which arbitrage is available and one has a strong belief about the likelihood of each event. In such situations, this method will maximize your expected profit.

Let $\vec d \in \mathbb{R}^n$ be our prespecified probability distribution so that $d_i$ holds the probability of outcome $i$ occurring. Knowing that

$$
E[\text{Payout}]=\vec d ^\intercal (A \vec w) = (\vec d ^\intercal A) \vec w, \notag
$$

we can easily write our objective as

$$
\begin{align}
\text{Maximize: }& (\vec d ^\intercal A) \vec w \\
\text{such that:}& \notag \\
& \vec 1 ^\intercal \vec w = 1 \notag \\
& \vec w \geq 0 \notag \\
& A \vec w \geq 1. \notag
\end{align}
$$

The output of a formulation such as this is somewhat useful, however thinking about probability distributions and $E[\text{profit}]$ becomes far more interesting when we discuss the use of quadratic programming to minimize variance. More on this below.

### The Existence Of A Solution

Taking a step back and shifting to a geometric perspective leads to an obvious but interesting conclusion: the existence of a solution depends entirely on the constraints. Specifically, the constraints specify a $b$-dimensional region (called the feasible region) in which $\vec w$ could live and still satisfy each constraint. If such a region doesn't exist, then clearly the problem has no solution.

The point is that $\text{Problem #1}$ and $\text{Problem #3}$ have the same constraints, and thus if one of them has a solution then both of them have a solution. The difference between them is which direction we need to move $\vec w$ within the feasible region to maximize the given objective. As such, once we establish our constraints there is freedom to choose an objective that suits our needs. Of course there is always the possibility that the feasible region is a point, which would lead all objectives to have the same solution.

#### Equal-Payout Arbitrage Is Rare

Generally speaking (i.e. given no specific form for $A$) we see that $\text{Problem #2}$ has additional equality constraints such that in practice it is far less likely to find (profitable) arbitrage in this situation. As a result, it is beneficial to look for this type of arbitrage only after finding arbitrages of the other form(s).

These additional equality constraints specify a region of its own, which we will call $\text{Region X}$. Referring to the region generated by either $\text{Problem #1}$ or $3$ as $\text{Region Y}$, then $\text{Problem #2}$ will only have a profitable arbitrage opportunity if some part of $\text{Region X}$ overlaps with $\text{Region Y}$, or $\text{Region X} \cap \text{Region Y} \neq \emptyset$.

It goes without saying that certainty in payouts is desirable, which is why an equal-payout arbitrage is worth seeking. But clearly there is only arbitrage inside $\text{Region Y}$, so perhaps there is a better way to go about our goal of minimizing variance? In fact, there is: quadratic programming. 

### Quadratic Programming

With quadratic programming, we can start in $\text{Region Y}$ and minimize variance from there. If there is a profitable equal-payout arbitrage then we will find it. If there is not, then we will find the next best thing. Quadratic programming, in the form we will discuss, is quite similar to linear programming but instead of having a linear objective function we have a quadratic objective function.

Specifically, we seek to minimize variance. Let $\Sigma \in \mathbb{R}^{b \times b}$ denote the covariance matrix of our bet payouts under the distribution $\vec d$ discussed in $\text{Problem #3}$. Knowing that the total portfolio variance is given by $\vec w ^\intercal \Sigma \vec w$, then our objective is

$$
\begin{align}
\text{Minimize: }& \vec w ^\intercal \Sigma \vec w \\
\text{such that:}& \notag \\
& \vec 1 ^\intercal \vec w = 1 \notag \\
& \vec w \geq 0 \notag \\
& A \vec w \geq 1 \notag \\
& (\vec d ^\intercal A) \vec w \geq r^*, \notag
\end{align}
$$

where $$r^*$$ is some arbitrary expected return threshold. The idea is that *given* some defined level of return, we want to find the $\vec w$ that *minimizes* the portfolio variance. If we did not include the $r^*$ constraint then we would always just find the global minimum variance portfolio. By including it, we allow ourselves the opportunity to choose exactly what we are looking for.

__Note:__ In fact, there may be multiple minimum variance portfolios. Given our general form for $A$, we can ensure that $\Sigma$ is positive semi-definite (just like any covariance matrix) but not necessarily positive definite. This is because we might have some bet(s) that is a linear combination of other bets. If $\Sigma$ is positive semi-definite but not positive definite, then we will have multiple solutions to $(4)$. Thus, by including $r^*$ we can seek the best possible portfolio.
{:.info}

Realistically, we would set a range of values for $$r^*$$ and compare the expected return with the variance for each. As such, it is useful to have some bounds on the values to compute. First, we know that an arbitrage defined by $\text{Region Y}$ is always profitable. Second, we know that the maximum expected return we could possibly earn with arbitrage would be given by the output of $\text{Problem #3}$. If we call the optimum for $\text{Problem #3}$ $r_\text{max}$, then we know that $0 \leq r^* \leq r_\text{max}$.
Any algorithm computing across this range could start at $r^*=0$ and then immediately jump to whatever expected return was found in that first step, and then carry on from there until $r_\text{max}$.

#### The Prior Distribution

An evident downfall of this approach is the extreme difficulty in properly establishing a probability distribution for the outcomes. If you were able to predict the probabilities with any decent accuracy you would already be an extremely profitable sports-bettor. Still, this approach is quite beneficial even under basic assumptions. One option would be to assume a completely uniform distribution -- in which case we simply want to seek even payouts. Another option could be to use the implied probabilities present in the odds, take some sort of average by outcome, and then scale everything so that $\vec d$ sums to one.


### Python Implementation
Below, we provide some basic code for both the linear programming and quadratic programming problems. We use `scipy` for the the former and `cvxopt` for the latter.

#### Linear Programming ($\text{Problem #2}$)

```python
import numpy as np
from scipy.optimize import linprog

def complex_arb_equal(A, notional=100, j=0, return_outcomes=True):
    '''
    Given matrix A of payouts, calculates an arbitrage the maximizes the profit in the event of outcome j.
    Note that j is indexed starting from 0.
    '''
    
    # Dimensions (n: number of outcomes, b: number of available bets)
    n, b = A.shape
    
    # Objective (WLOG we can always let this be the first outcome, but we'll leave j in for clarity purposes)
    obj = -A[j]
    
    ### BUILDING THE CONSTRAINTS ###
    
    # Equality constraint: weights sum to 1
    matrix_eq_1 = np.ones((1, b))
    constr_eq_1 = np.array(1)
    
    # Equality constraint: (A_-j - J) w = 0
    J = np.concatenate([A[j]] * (n - 1))
    A_minus_j = A[np.arange(n) != j]
    
    matrix_eq_2 = A_minus_j - J
    constr_eq_2 = np.zeros(n - 1)
    
    # Combining equality constraints
    matrix_eq = np.concatenate([matrix_eq_1, matrix_eq_2])
    constr_eq = np.append(constr_eq_1, constr_eq_2)

    # Inequality constraints: no potential losses
    matrix_ineq = -A
    constr_ineq = - np.ones(n)

    # Calculate the weights. By default the bounds are (0, None) but we add it anyway to emphasize that the w >= 0
    prog = linprog(c = obj, A_eq = matrix_eq, b_eq = constr_eq, A_ub = matrix_ineq, b_ub = constr_ineq, bounds=(0, None))
    
    # Check the status. Typically we will only encounter status 0 (success) or 2 (problem is unfeasible/no solution)
    # For now, if we get another status we will just alert the user.
    if prog.status == 2:
        print('No solution was found. Arbitrage is not feasible.')
        return
    
    elif prog.status != 0:
        raise Exception(f"Status {prog.status}: {prog.message}")
        
    # If arb was found, extract the bet weights, and the returns in each situation
    w_vec = prog.x.round(4)
    returns = ((A @ prog.x) - 1)#.round(4)
    
    if return_outcomes:
        return notional * w_vec, returns
    
    else:
        return notional * w_vec
```

#### Quadratic Programming
First, we construct a basic function to calculate the covariance matrix, $\Sigma$, given $A$. Then we implement the quadratic program.

```python
from cvxopt import cvxopt
from cvxopt.solvers import qp
from cvxopt import matrix
cvxopt.solvers.options["show_progress"] = False
cvxopt.solvers.options["abstol"] = 1e-10

# Covariance Matrix
def calculate_sigma(A, d_vec):
    
    n, b = A.shape
        
    mu_vec = A.T @ d_vec
    sigma = 0
    
    for i in range(n):
        sigma += d_vec[i] * np.outer(A[i] - mu_vec, A[i] - mu_vec)
    
    return sigma

# Quadratic Programming
def complex_arb_minvar(A, r_star=0, d_vec=None, notional=100, return_outcomes=True):
    
    # Dimensions (n: number of outcomes, b: number of available bets)
    n, b = A.shape
    
    # Uniform by default
    if d_vec is None:
        d_vec = np.ones(n) / n
    
    # Covariance matrix
    sigma = calculate_sigma(A, d_vec)
    
    # Reshape to please CVXOPT
    d_vec = d_vec.reshape((n, 1))
    
    # Objective Function. CVXOPT requires an input for the second AND first degree, but the first degree is 0 for this.
    P = sigma
    q = np.zeros(b)
    
    ### BUILDING THE CONSTRAINTS ###
    
    # Equality constraint: weights sum to 1
    matrix_eq = np.ones((1, b))
    constr_eq = [1]

    # Inequality constraints: no potential losses
    matrix_ineq_1 = -A
    constr_ineq_1 = -np.ones(n)
    
    # Inequality constraints: minimum expected return
    matrix_ineq_2 = -d_vec.T @ A
    constr_ineq_2 = -(1 + r_star)
    
    # Inequality constraints: all weights are positive
    matrix_ineq_3 = -np.eye(b)
    constr_ineq_3 = np.zeros(b)
    
    # Combine them
    matrix_ineq = np.concatenate([matrix_ineq_1, matrix_ineq_2, matrix_ineq_3])
    constr_ineq = np.concatenate([np.append(constr_ineq_1, constr_ineq_2), constr_ineq_3])
    
    # Convert to CVXOPT's matrix class of type double
    P, matrix_eq, matrix_ineq = matrix(P, tc='d'), matrix(matrix_eq, tc='d'), matrix(matrix_ineq, tc='d')
    q, constr_eq, constr_ineq = matrix(q, tc='d'), matrix(constr_eq, tc='d'), matrix(constr_ineq, tc='d')
    
    # Calculate the weights. By default the bounds are (0, None) but we add it anyway to emphasize that the w >= 0
    try:
        prog = qp(P=P, q=q, G=matrix_ineq, h=constr_ineq, A=matrix_eq, b=constr_eq)
    except ValueError as e:
        print("Problem not feasible! CVXOPT domain error.")
        return
    
    # Check the status
    
    if prog['status'] != 'optimal':
        print('No solution was found. Arbitrage is not feasible.')
        return
        
    # If arb was found, extract the bet weights, and the returns in each situation
    w_vec = np.array(prog['x'])
    returns = ((A @ w_vec) - 1)
    std_dev = np.sqrt((w_vec.T @ sigma @ w_vec).round(4))[0][0]
    
    if return_outcomes:
        return notional * w_vec.round(4), std_dev.round(4), returns.round(4)
    
    else:
        return notional * w_vec.round(4), std_dev.round(4)
```