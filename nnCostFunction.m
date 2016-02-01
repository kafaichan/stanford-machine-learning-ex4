function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
    a1 = [ones(m,1) X]; % dim = [5000*401]
    z2 = a1 * Theta1'; % dim = [5000*401]*[25*401]' = [5000*25]
    a2 = sigmoid(z2); % dim = [5000*25]
    a2 = [ones(m,1) a2]; % dim = [5000*26]
    z3 = a2 * Theta2'; % dim = [5000*26] * [10*26]' = [5000*10]
    a3 = sigmoid(z3); % dim = 5000 * 10
    
    boolean_matrix = zeros(m ,num_labels);
    for i = 1:num_labels
        boolean_matrix(:,i) = (y == i);
    end
    J = (-1/m) * sum(sum(boolean_matrix .* log(a3) + (1-boolean_matrix).* log(1-a3)));
    
    nobias_Theta1 = [zeros(hidden_layer_size,1) Theta1(:,2:end)];
    nobias_Theta2 = [zeros(num_labels,1) Theta2(:,2:end)];
    nobias_ThetaVec = [nobias_Theta1(:);nobias_Theta2(:)];
    
    J = J + (lambda/(2*m))* (nobias_ThetaVec'*nobias_ThetaVec);
    
    
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% vectorize method(faster than for-loop :D)

delta3 = a3 - boolean_matrix; % dim = [5000*10], each sample per row
delta2 = (delta3 * Theta2(:,2:end)) .* sigmoidGradient(z2); % dim = [5000*10]*[10*26](2:end) ->[5000*25] .* [5000*25] =>[5000*25]

Theta1_grad = delta2' * a1; %dim = [5000*25]'*[5000*401] = [25*401]
Theta2_grad = delta3' * a2; %dim = [5000*10]'*[5000*26] = [10*26]

Theta1_grad = (1/m)*Theta1_grad;
Theta2_grad = (1/m)*Theta2_grad;

% not vectorize method
%    for i = 1:m
%        a1 = X(i,:); % dim = 1*401
%        z2 = a1 * Theta1'; % dim = [1*401] * [25*401]'
%        a2 = sigmoid(z2); % dim = [1*25]
%        a2 = [ones(1,1) a2]; % dim = [1*26]
%        z3 = a2 * Theta2'; % dim = [1*26] * [10*26]'
%        a3 = sigmoid(z3); % dim = [1*10]
%        
%        delta3 = a3 - boolean_matrix(i,:); % dim = [1*10], yk = 0/1
        
%        delta2 = delta3 * Theta2 .* [1 sigmoidGradient(z2)]; % dim = [1*10] * [10*26] .* [1*26] = [1*26]
%        Theta1_grad = Theta1_grad + (delta2(2:end)'* a1); % dim = [1*25]' * [1*401]
%        Theta2_grad = Theta2_grad + (delta3' * a2); % [1*10]'* [1*26];
%    end
   
%    Theta1_grad = 1/m .* Theta1_grad;
%    Theta2_grad = 1/m .* Theta2_grad;
    
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

grad = (lambda/m)*nobias_ThetaVec + grad;

% -------------------------------------------------------------

% =========================================================================





end
