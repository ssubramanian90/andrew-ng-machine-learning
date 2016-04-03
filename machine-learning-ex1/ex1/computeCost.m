function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
h=X*theta;
J=sum((h-y).^2)/2/m;


grad(1,1) = (1/m)*sum((h-y).*X(:,1)); 
grad(2:end,1)=((1/m)*((h-y)'*X(:,2:end)))'+(lambda/m)*theta(2:end);


% =========================================================================

end
