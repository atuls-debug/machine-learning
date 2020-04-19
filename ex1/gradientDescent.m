function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
to_calc_theta = zeros(2,1)
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    to_calc_theta = alpha * (1/m) * (sum(((X * theta) - y) .* X));
    %fprintf('to_calc_theta=%f \n',to_calc_theta);
    %theta = theta - (alpha * (1/m) * (sum(((X * theta) - y) .* X)));
    theta = theta - transpose(to_calc_theta);
    %fprintf('Iter=%d thetaValue = %f %f \n', iter,theta(1),theta(2));
    %fprintf('J History value = %f\n\n',J_history(iter));
   
    % ============================================================

    % Save the cost J in every iteration    
    J_history = computeCost(X, y, theta);
    %fprintf('J History value = %f\n\n',J_history); 
end

end
