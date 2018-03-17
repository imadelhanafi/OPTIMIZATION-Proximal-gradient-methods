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

### Algorithm



## Part 2 : Cutting-plane algorithm & Bundle methods


----- 
