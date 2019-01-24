Linear Regression

My implementation of linear regression was straightforward, I simply converted the mathematical
equations that defined all three methods into code. I raised the number of epochs for stochastic
gradient descent to 1500, and in my tests I played with the learning rate to ensure that all three
methods converged. All three methods performed fairly well on the data.

Logistic Regression

I made numerous changes to this script. After implementing the equations as written, I found that I
was getting numerous overflow errors due to the exponential function returning a result that was too
large. So I modified the sigmoid function to ensure that the exponential always had a negative exponent.
The prediction code used the sigmoid function, so I changed the prediction threshold to simply check
if the dot product of the weights and a data point was above zero. This is mathematically equivalent
to checking if the sigmoid function is above 0.5. There was also an overflow in my implementation
of the average log likelihood, since there is an exponential term in there as well. I chose to
handle overflows by approximating log(1+exp(xT * beta)) with xT * beta, since if an overflow happens
that means that the xT*beta term contributes the most to the log term.

I implemented regularization for batch gradient descent by adding a boolean method that toggles the
feature. By default regularization is on. My implementation of batch gradient descent uses the
equations presented in the lecture slides. I set the learning rate to be different depending on whether
normalization is used or not. My Newton-Raphson method implementation also uses the equations presented
in the lecture slides. I found that with the non-normalized data the code threw a LinAlgError because
the hessian matrix was not invertible. I handled this exception by subtracting a small portion of the identity
matrix to the hessian matrix to make it invertible.

Testing both linear and logistic regression may require some fiddling with the learning rate, you can run
them both with
./<scriptname> <method number> <normalization flag>
