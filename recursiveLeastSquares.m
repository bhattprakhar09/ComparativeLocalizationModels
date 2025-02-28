function [mu, rmse_rls, resnorm_values, rmse_values] = recursiveLeastSquares(anchorPositions2D, observedRanges, tgtpos2D, initial_estimate)
    mu = initial_estimate;
    tolerance = 1e-5;
    maxIters = 100;
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
        %damped pseudoinverse
        lambda = 0.1; 
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