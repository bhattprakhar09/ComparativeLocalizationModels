function [mu, rmse_rls, resnorm_values, rmse_values] = recursiveLeastSquares(anchorPositions2D, observedRanges, tgtpos2D, initial_estimate)
% RECURSIVELEASTSQUARES Performs a simple iterative least-squares localization.
%
%   This function refines an initial position estimate (mu) by iteratively
%   minimizing the difference between the observed ranges (observedRanges)
%   and the ranges implied by the current estimate of mu.
%
%   Inputs:
%     anchorPositions2D : 2 x nAnchors  array of anchor coordinates [x; y]
%     observedRanges    : 1 x nAnchors  measured distances from each anchor
%     tgtpos2D          : 2 x 1         true target position (for RMSE calc)
%     initial_estimate  : 2 x 1         starting guess for the target position
%
%   Outputs:
%     mu            : 2 x 1  final estimated position after convergence
%     rmse_rls      : scalar final RMSE (distance) between mu and tgtpos2D
%     resnorm_values: 1 x k  residual norms at each iteration (k = #iterations)
%     rmse_values   : 1 x k  RMSE at each iteration

    mu = initial_estimate; % Current estimate of position
    tolerance = 1e-5; % Convergence threshold
    maxIters = 100; % Max iterationsS
    numAnchors = size(anchorPositions2D, 2);

    resnorm_values = zeros(1, maxIters);
    rmse_values = zeros(1, maxIters);

    for iteration = 1:maxIters
        % Calculate residuals
        measuredRanges = sqrt(sum((mu - anchorPositions2D).^2, 1));
        r = observedRanges - measuredRanges;
        % disp(['Size of observedRanges: ', num2str(size(observedRanges))]);
        % disp(['Size of measuredRanges: ', num2str(size(measuredRanges))]);


        % Residual norm
        resnorm = sum(r.^2);
        %disp(r)
        resnorm_values(iteration) = resnorm;

        % Jacobian
        H = zeros(numAnchors, 2);
        for i = 1:numAnchors
            dx = mu(1) - anchorPositions2D(1, i);
            dy = mu(2) - anchorPositions2D(2, i);
            d_i0 = measuredRanges(i);
            H(i, :) = [dx / d_i0, dy / d_i0];
        end

        % Update position
        %delta = pinv(H' * H) * (H' * r');
        %damped pseudoinverse to correctly hadle ill conditioned matrix
        lambda = 0.001; 
        delta = (H' * H + lambda * eye(2)) \ (H' * r');
        mu = mu + delta;

        % RMSE calculation
        diff_rls = tgtpos2D - mu;
        rmse_values(iteration) = sqrt(mean(diff_rls.^2));
        % disp(['RMSE at Iteration ', num2str(iteration), ': ', num2str(rmse_values(iteration))]);

        % Check convergence
        if norm(delta) < tolerance || (iteration > 1 && abs(rmse_values(iteration) - rmse_values(iteration-1)) < tolerance)
            % disp(['Converged after ', num2str(iteration), ' iterations']);
            resnorm_values = resnorm_values(1:iteration);
            rmse_values = rmse_values(1:iteration);
            break;
        end
    end

    % % Plot residual norm
    % figure;
    % plot(1:length(resnorm_values), resnorm_values, '-o');
    % xlabel('Iteration');
    % ylabel('Residual Norm');
    % title('Residual Norm in Recursive Least Squares');
    % grid on;
    % 
    % % Plot RMSE over iterations
    % figure;
    % plot(1:length(rmse_values), rmse_values, '-o');
    % xlabel('Iteration');
    % ylabel('RMSE');
    % title('RMSE vs Iterations in Recursive Least Squares');
    % grid on;

    % Final RMSE
    rmse_rls = rmse_values(end);
end