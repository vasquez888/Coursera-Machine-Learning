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

% X is 12 x 1
% y is 12 x 1
% X = [ones(m, 1), X(:,1)]; % Add a column of ones to X
h = X * theta;
thr = theta(2:length(theta), 1);
J = (1 / (2*m)) * ((h) - y)' * ((h) - y) + (lambda / (2*m)) * sum(thr.^2);
grad = (1/m) * (X' * ((h) - y) + lambda * theta);
grad(1) = (1/m).*sum((h-y).*X(:,1));

% =========================================================================

grad = grad(:);

end
