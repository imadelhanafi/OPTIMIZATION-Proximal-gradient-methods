# Non-differentiable optimization - Proximal Methods

## Part 1 : ISTA and FISTA on the LASSO Problem

### Problem

In regression analysis, Least Absolute Shrinkage and Selection Operator (LASSO) allows us to do variable selection (features selection) in order to get the most relevant features. 
In this project we will solve the optimisation problem associated to LASSO using proximal method algorithm (simple and accelerated form).

Let's consider the following model 

![equation](http://latex.codecogs.com/svg.latex?Y=Xu+\epsilon )

Where ![equation](http://latex.codecogs.com/svg.latex?Y\in\mathds{R}^n,X\in\mathds{R}^{n\times%20p}) and ![equation](http://latex.codecogs.com/svg.latex?u\in\mathds{R}^{p})

In practice, the labels  ![equation](http://latex.codecogs.com/svg.latex?X_{i})
 (columns of the matrix X) are not all relevant. It is necessary to eliminate *unnecessary* features (variables). The idea of LASSO is therefore not to make a classical linear regression but a regularized regression which forces certain coefficients of the estimated estimate  *u* to be set to zero.

Therefore, the optimization problem to be solved can be written as 

![equation](https://latex.codecogs.com/svg.latex?%5Chat%7Bu%7D%28%5Clambda%29%20%3A%3D%5Coperatorname*%7Bargmin%7D_%7Bu%5Cin%5Cmathds%7BR%7D%5E%7Bp%7D%7D%201/2%20%5C%7CY%20-%20Xu%5C%7C_%7B2%7D%5E%7B2%7D%20&plus;%20%5Clambda%5C%7Cu%5C%7C_%7B1%7D)

λ is the regularization constant.

### Algorithm of proximal gradient

The above optimization problem can be seen as (1) : 

![equation](https://latex.codecogs.com/svg.latex?\operatorname*{min}_{u\in\mathds{R}^{p}}F(u)+G_{\lambda}(u))

Where F is the sqaure of 2-norm, which is clearly convex, differentiable with a Lipschitzien gradient of constant L. 
We define the proximal operator of the function ![equation](https://latex.codecogs.com/svg.latex?\epsilo%20G_\lambda) as 

![equation](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BP%7D_%7B%5Cvarepsilon%20G_%7B%5Clambda%7D%7D%28u%29%20%26%20%3A%3D%5Coperatorname*%7Bargmin%7D_%7Bv%5Cin%20%5Cmathds%7BR%7D%5E%7Bp%7D%7D%20%5Cvarepsilon%20G_%7B%5Clambda%7D%20%28v%29&plus;%201/2%5C%7Cv-u%5C%7C_%7B2%7D%5E%7B2%7D)

If u# is a solution to the problem (1), the optimality conditions can be written as 

![equation](https://latex.codecogs.com/svg.latex?%5Cexists%5C%3B%20r%5E%7B%5C%23%7D%20%5Cin%20%5Cpartial%20G_%7B%5Clambda%7D%28u%5E%7B%5C%23%7D%29%2C%20%5Ctext%7B%20t.q.%20%7D%5C%3B%20r%5E%7B%5C%23%7D&plus;%20%5Cnabla%20F%28u%5E%7B%5C%23%7D%29%20%3D%200)

By the definition of the proximal operator, this is equivalent to 

![equation](https://latex.codecogs.com/svg.latex?u%5E%7B%5C%23%7D%20%3D%20%5Cmathcal%7BP%7D_%7B%5Cvarepsilon%20G_%7B%5Clambda%7D%7D%28u%5E%7B%5C%23%7D%20-%20%5Cvarepsilon%5Cnabla%20F%28u%5E%7B%5C%23%7D%29%29)

The algortihm then consists on finding a fixed point that satistifes the last equation. Since the proximal operator of ![equation](https://latex.codecogs.com/svg.latex?\epsilo%20G_\lambda) is easy to compute, an iterative algorthim leads to a fixed point.

- Choose an initial point
- Do until convergence: 
  - choose a step 
  - ![equation](https://latex.codecogs.com/svg.latex?u%5E%7Bk&plus;1%7D%20%3D%20%5Cmathcal%7BP%7D_%7B%5Cvarepsilon%5E%7B%28k%29%7DG_%7B%5Clambda%7D%7D%28u%5E%7B%28k%29%7D-%5Cvarepsilon%5E%7Bk%7D%5Cnabla%20F%28u%5E%7B%28k%29%7D%29%29)
  
 ## FISTA 
 



## Part 2 : Cutting-plane algorithm & Bundle methods


----- 
