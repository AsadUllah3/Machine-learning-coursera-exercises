function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X=12*2  theta 2x1 X
pred=X*theta;  % size 12*1
v=pred-y;
a1=sum(v.^2);
a=a1/(2*m);
theta(1)=0;
nonreg= (lambda/(2*m)) .*(sum(theta.^2));
J=a+nonreg;


[a,b]=size(grad);
 grad(1)=((1/m) .* sum(((pred)-y) .* X(:,1)));
for i=2:a,
 grad(i)=(((1/m) .* sum(((pred)-y) .* X(:,i)))+((lambda .* theta(i))/m));













% =========================================================================

grad = grad(:);

end
