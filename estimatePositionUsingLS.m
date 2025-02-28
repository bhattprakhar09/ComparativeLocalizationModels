function [initial_estimate, rmse, intersection_areas] = estimatePositionUsingLS(anchorPositions2D, estimated_ranges, tgtpos2D)
% Inputs:
%    anchorPositions2D - 2xN matrix of anchor positions [x; y]
%    estimated_ranges  - 1xN vector of estimated distances from each anchor
%    tgtpos2D          - 2x1 true target position [x; y]
%
% Outputs:
%    initial_estimate  - LS estimated target position
%    rmse              - Root mean square error of the estimate
%    intersection_areas- Matrix containing pairwise circle intersection areas
%
    numAnchors = size(anchorPositions2D, 2);

    % LS formulation
    A = zeros(numAnchors - 1, 2);
    b = zeros(numAnchors - 1, 1);
    refAnchorPos = anchorPositions2D(:, 1);
    refRange = estimated_ranges(1);

    for i = 2:numAnchors
        xi = anchorPositions2D(1, i);
        yi = anchorPositions2D(2, i);
        ri = estimated_ranges(i);
        A(i-1, :) = [2 * (xi - refAnchorPos(1)), 2 * (yi - refAnchorPos(2))];
        b(i-1) = (refRange^2 - ri^2) - (refAnchorPos(1)^2 - xi^2) - (refAnchorPos(2)^2 - yi^2);
    end

    % Calculate initial estimate
    initial_estimate = A \ b;
    % disp('Initial position estimate using least squares:');
    % disp(initial_estimate);

    % Plot results
    figure;
    hold on;

    % % Plot anchors and their circles
    % theta = linspace(0, 2*pi, 200); % Higher resolution for smoother circles
    % for i = 1:numAnchors
    %     xi = anchorPositions2D(1, i);
    %     yi = anchorPositions2D(2, i);
    %     ri = estimated_ranges(i);
    %     x_circle = xi + ri * cos(theta);
    %     y_circle = yi + ri * sin(theta);
    %     plot(x_circle, y_circle, '--', 'LineWidth', 1.5, 'DisplayName', sprintf('Range from Anchor %d', i));
    % end
    % 
    % % Plot estimated position and true target
    % plot(initial_estimate(1), initial_estimate(2), 'mp', 'MarkerSize', 15, 'DisplayName', 'Estimated Position');
    % plot(tgtpos2D(1), tgtpos2D(2), 'g*', 'MarkerSize', 10, 'DisplayName', 'True Target');
    % 
    % % Annotate anchor positions
    % for i = 1:numAnchors
    %     text(anchorPositions2D(1, i), anchorPositions2D(2, i), ...
    %         sprintf('(%0.2f, %0.2f)', anchorPositions2D(1, i), anchorPositions2D(2, i)), ...
    %         'VerticalAlignment', 'top', 'HorizontalAlignment', 'center');
    % end
    % 
    % legend;
    % xlabel('X Coordinate');
    % ylabel('Y Coordinate');
    % title('Results for Estimated Position');
    % axis equal;
    % grid on;

    % Calculate RMSE
    diff = tgtpos2D - initial_estimate;
    rmse = sqrt(mean(diff.^2));
    % disp("RMSE for estimated position: ");
    % disp(rmse);
    % hold off;

    % Intersection area calculations
    intersection_areas = zeros(numAnchors);
    for i = 1:numAnchors
        for j = i+1:numAnchors
            r1 = estimated_ranges(i);
            r2 = estimated_ranges(j);
            d = norm(anchorPositions2D(:, i) - anchorPositions2D(:, j));
            intersection_areas(i, j) = calculateIntersectionArea(r1, r2, d);
        end
    end

    % % Display pairwise intersection areas
    % for i = 1:numAnchors
    %     for j = i+1:numAnchors
    %         fprintf('Intersection area between Anchor %d and Anchor %d: %.2f m^2\n', i, j, intersection_areas(i, j));
    %     end
    % end

    % Handle common area or line
    %handleCommonAreaOrLine(anchorPositions2D, estimated_ranges, numAnchors);
end
%%
function A = calculateIntersectionArea(r1, r2, d)
    if d >= (r1 + r2)
        A = 0; % No intersection
    elseif d <= abs(r1 - r2)
        A = pi * min(r1, r2)^2; % One circle inside the other
    else
        part1 = r1^2 * acos((d^2 + r1^2 - r2^2) / (2 * d * r1));
        part2 = r2^2 * acos((d^2 + r2^2 - r1^2) / (2 * d * r2));
        part3 = 0.5 * sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2));
        A = part1 + part2 - part3;
    end
end
%%
function handleCommonAreaOrLine(anchorPositions2D, estimated_ranges, numAnchors)
    % figure;
    % hold on;

    % Define grid for evaluating common area
    x_min = min(anchorPositions2D(1, :) - estimated_ranges);
    x_max = max(anchorPositions2D(1, :) + estimated_ranges);
    y_min = min(anchorPositions2D(2, :) - estimated_ranges);
    y_max = max(anchorPositions2D(2, :) + estimated_ranges);

    grid_resolution = max(0.1, min((x_max - x_min) / 200, (y_max - y_min) / 200));
    [x_grid, y_grid] = meshgrid(x_min:grid_resolution:x_max, y_min:grid_resolution:y_max);

    % Identify grid points within all anchors' ranges
    common_area_points = ones(size(x_grid));
    for i = 1:numAnchors
        xi = anchorPositions2D(1, i);
        yi = anchorPositions2D(2, i);
        ri = estimated_ranges(i);
        distances = sqrt((x_grid - xi).^2 + (y_grid - yi).^2);
        common_area_points = common_area_points & (distances <= ri);
    end

    % % Plot circles again for clarity
    % theta = linspace(0, 2*pi, 200);
    % for i = 1:numAnchors
    %     xi = anchorPositions2D(1, i);
    %     yi = anchorPositions2D(2, i);
    %     ri = estimated_ranges(i);
    %     x_circle = xi + ri * cos(theta);
    %     y_circle = yi + ri * sin(theta);
    %     plot(x_circle, y_circle, '--', 'LineWidth', 1.5);
    % end
    % 
    % % Plot the common intersection area
    % [rows, cols] = find(common_area_points);
    % scatter(x_grid(sub2ind(size(x_grid), rows, cols)), ...
    %         y_grid(sub2ind(size(y_grid), rows, cols)), 10, 'r', 'filled', ...
    %         'MarkerFaceAlpha', 0.5, 'DisplayName', 'Common Area');
    % 
    % % Finalize plot
    % legend show;
    % xlabel('X Coordinate');
    % ylabel('Y Coordinate');
    % title(sprintf('Intersection for %d Anchors', numAnchors));
    % axis equal;
    % grid on;
    % hold off;
end
