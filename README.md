# Sharpness and Large Step in Gradient Descent
A repository to catalogue studying and experimentation on EoS, Sharpness and large-step Gradient Descent.

Based on *Any-stepsize Gradient Descent for Separable Data under Fenchel-Young Losses* by Bao et al. (https://arxiv.org/abs/2502.04889), *Large Stepsize Gradient Descent for Logistic Loss: Non-Monotonicity of the Loss Improves Optimization Efficiency* by Wu et al. (https://arxiv.org/abs/2402.15926), and "Flatland: The Adventures of Gradient Descent with Large Step Sizes" by Galli et al. (https://arxiv.org/abs/2606.06722).

## 1. Large Step under Logistic Regression

We're studying the behavior of large step-sizes on Gradient Descent (GD), one of the most common optimizers in Machine Learning. 

Wu et al. shows that under logistic regression with linearly separable data, GD with large step sizes still converges. Bao et al. extends this finding further. We will start by analyzing the simpler case in Wu et al, and then looking to generalize this.


## Context

Wu et al. considers large GD step-sizes for binary classification with a linear model trained with Logistic Loss.

First, we need to understand what the classic theory says about how GD behaves.

For a $\beta$-smooth function, the descent lemma guarantees convergence if $0 < \eta < \cfrac{2}{\beta}$, where $\eta$ is the GD step-size.

We will study the behavior of large GD step-sizes with the same conditions of Wu et al., which defines the following:
- All data points $x_i$ have a magnitude of at most 1.
- The two classes are linearly separable.
- Labels are defined as $y_i \in \{-1, +1\}$
- There is a unit vector $w^{\ast}$ and a $\gamma > 0$ such that $\langle y_i x_i, w^{\ast}\rangle \geq \gamma$ for all $i$

This last point is used to refer to the fact that under an optimal vector $w^{\ast}$, the closest point to the decision boundary is at distance $\gamma > 0$.
This means that the vector $\frac{w^{\ast}}{\gamma}$ produces a score $z_i = y_i \langle x_i, \frac{w^{\ast}}{\gamma}\rangle \geq 1$ for all $i$.

<img src="https://github.com/slamgLuke/anystep-gradient-descent/raw/main/boards/board1.jpg" width="500">




## Logistic Loss Landscape

Optimizing the model is equivalent to maximizing the likelihood for the probability function:

$$p(y_i|x_i) = \sigma(y_i \langle x_i, w\rangle)$$
$$\text{where  } \sigma(z) = \cfrac{1}{1 + e^{-z}}$$

$$\arg\max_w\text{  } \prod_{i=1}^{n}p(y_i|x_i)$$

Which is equivalent to minimizing the average Negative Log-likelihood.

$$\mathcal{L}(w) = \frac{1}{n}\sum_{i=1}^{n}-\log p(y_i|x_i)$$

Now we will derive to find the $\beta$-smoothness of $\ell(z_i) = -\log p(y_i|x_i)$, and then use that to find the smoothness for $\mathcal{L}(w)$.

<img src="https://github.com/slamgLuke/anystep-gradient-descent/raw/main/boards/board2.jpg" width="500">

The second derivative of $\ell(z)$ peaks at $z=0$, with a value of 1/4.

Since 1/4 is the absolute largest value of $\ell''(z)$, we can use the Mean Value Theorem to show that $\ell'(z)$ is 1/4-Lipschitz.

And the definition for smoothness states that a function is $\beta$-smooth when its derivative is $\beta$-Lipschitz. $\ell(z)$ is 1/4-smooth.

Now we bring this to find the smoothness of $\mathcal{L}(w)$:

<img src="https://github.com/slamgLuke/anystep-gradient-descent/raw/main/boards/board3.jpg" width="500">

Because $y_i$ values are exactly +1 or -1, and the data points $x_i$ are bounded to have a norm of 1 at most, the gradient of the NLL cannot change at a larger rate than the derivative of the log probability when varying the weights. Hence $\nabla\mathcal{L}(w)$ is 1/4-Lipschitz, and therefore, $\mathcal{L}(w)$ is 1/4-smooth.

The descent lemma guarantees GD convergence for any $0 < \eta < 8$.


## Edge of Stability (EoS)

Cohen et al. (https://arxiv.org/abs/2103.00065) states how Gradient Descent training becomes unstable when the sharpness value (max eigenvalue of the Loss Hessian) hovers at or surpasses $\frac{2}{\eta}$. This threshold is called Edge of Stability (EoS), where GD causes the loss to oscillate instead of monotonically dropping.

We will explore this definition in the context of the Logistic Loss, and find a bound to its sharpness value.

<img src="https://github.com/slamgLuke/anystep-gradient-descent/raw/main/boards/board4.jpg" width="500">

The Hessian is defined by the outer product of each data point $\mathbf{x}_i$ by itself: $\mathbf{x}_i\mathbf{x}_i^\top$, scaled by the second derivative of the score $z_i$, and averaged for all $i$ data points.

<img src="https://github.com/slamgLuke/anystep-gradient-descent/raw/main/boards/board5.jpg" width="500">

Max eigenvalue properties give us an upper bound for the sharpness of the Logistic Loss, which is 1/4. Smoothness and Edge of Stability in GD are very closely related topics.

Following this logic, EoS can only theoretically be reached if a value of $\eta \geq 8$ is used, which is outside the bounds of the descent lemma.

Let's evaluate under what conditions this 1/4 bound is reached:

<img src="https://github.com/slamgLuke/anystep-gradient-descent/raw/main/boards/board6.jpg" width="500">

So, when initializing the weights at $w_0 = 0$, the maximum sharpness value is reached.

Since $x_ix_i^\top$ is constant, as $w$ moves away from 0, the value of $\ell''(z)$ decreases, and so does the sharpness.

## Convergence for large step-size

The expected behavior one would find from this is that when using a large $\eta$ value for training, the loss curve would me more unstable at the beginning, becoming more stable as $|w|$ grows.

However, we haven't explored GD convergence outside the range from the descent lemma. This is the key finding of Wu et al.

Wu et al. argues that GD with a large step-size **starts in an EoS phase**, before eventually transitioning into a **stable phase**.

Given our findings, we know that a starting $w_0 = 0$ can generate a sharpness value of up to 1/4, so the starting EoS phase would be true for $\eta \geq 8$ assuming the highest eigenvalue possible.

### Bound for Sharpness in a single point $w$

First, by showing that the sharpness at point w is bounded by the loss function $S(w) \leq L(w)$, we can show that as the loss decreases, GD moves away from EoS into a more stable phase.

The proof consists in showing that $\ell''(z) < \ell(z)$ for all $z$.

<img src="https://github.com/slamgLuke/anystep-gradient-descent/raw/main/boards/board7.jpg" width="500">

$$\ln (1 + e^{-z}) \geq \sigma(z)\sigma(-z)$$

$$\ell(z) \geq \ell''(z)$$

Hence $L(w)$ is an upper bound for the sharpness $S(w)$.

Joining this with $\ell''(z) \leq \frac{1}{4}$, we get in turn:

$$S(w) \leq \min(\frac{1}{4}, L(w))$$

<br>
<br>

While this is very useful, in order to prove convergence, we need for the bound to hold for a whole segment in a GD step-iteration.


### Bound for a segment $[w_t, w_{t+1}]$

Recall the GD definition:

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

After defining $\Delta = -\eta \nabla L(w_t)$, the variation in a single step, one can restrict $L$ to the segment between $w_t$ and $w_{t+1}$. Let's define an interpolation $\phi(\alpha)$ between these two:

$$\phi (\alpha) = L(w_t + \alpha\Delta)$$

$$\alpha \in [0, 1]$$

$$\phi(0) = L(w_t)$$
$$\phi(1) = L(w_{t+1})$$

Applying the Fundamental Theorem of Calculus, we can define the behavior of the segment between$\phi(0)$ and $\phi(1)$

<img src="https://github.com/slamgLuke/anystep-gradient-descent/raw/main/boards/board8.jpg" width="500">

As it is shown, if we apply an upper bound $\phi''(\beta) \leq M$ for all $\beta \in [0, 1]$, we get:

$$\phi(1) - \phi(0) = L(w_{t+1}) - L(w_t) \leq \frac{1}{n}\sum_{i=1}^n \ell'(z_i) \cdot y_i \langle \Delta, y_i\rangle  + \frac{M}{2}$$

Using the prior results: $\ell''(z) \leq min(\frac{1}{4}, \ell(z))$, as well as Cauchy-Schwarz, we can set an upper bound:

<img src="https://github.com/slamgLuke/anystep-gradient-descent/raw/main/boards/board9.jpg" width="500">

Since $\frac{||\Delta||^2}{4}$ does not depend on any $\beta$, this is an upper bound for the whole segment.

We can also represent the $\phi'(0)$ part in terms of $\nabla L(w)$.

<img src="https://github.com/slamgLuke/anystep-gradient-descent/raw/main/boards/board10.jpg" width="500">

## Generalization beyond Logistic

Wu et al. 
