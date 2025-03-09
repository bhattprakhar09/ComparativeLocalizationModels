function [estPosLS, rmseVec, intersection_areas_cell] = estimatePositionUsingLS_batchVectorized(anchorPositions2D, rangeMatrix, tgtposMatrix)
% ESTIMATEPOSITIONUSINGLS_BATCHVECTORIZED
% Performs a vectorized Least Squares (LS) localization for multiple points.
%
% Inputs:
%   anchorPositions2D : 2 x nBS    (each column is [x_b; y_b] for one anchor)
%   rangeMatrix       : nPoints x nBS  (rangeMatrix(i,b) => range from anchor b to point i)
%   tgtposMatrix      : 2 x nPoints    (optional: if provided, used for RMSE calculation)
%
% Outputs:
%   estPosLS               : 2 x nPoints  (the LS position estimates)
%   rmseVec                : 1 x nPoints  (RMSE for each point, if tgtposMatrix given)
%   intersection_areas_cell: cell array of intersection areas (optional usage)

    % --- Check input dimensions ---
    [nPoints, nBS] = size(rangeMatrix);
    if size(anchorPositions2D,2) ~= nBS
        error('Mismatch: anchorPositions2D has %d anchors, but rangeMatrix has %d columns.', ...
            size(anchorPositions2D,2), nBS);
    end
    
    if nargin < 3 || isempty(tgtposMatrix)
        tgtposMatrix = [];
    else
        if size(tgtposMatrix,2) ~= nPoints
            error('Mismatch: tgtposMatrix has %d points, but rangeMatrix has %d rows.', ...
                size(tgtposMatrix,2), nPoints);
        end
    end

    % We need at least 3 anchors for a 2D LS solution
    if nBS < 3
        error('At least 3 anchors are required for 2D LS. Found nBS=%d', nBS);
    end

    %% 1) Select the first anchor as reference
    refPos = anchorPositions2D(:,1);   % [xRef; yRef]
    xRef   = refPos(1);
    yRef   = refPos(2);

    %% 2) Build matrix A => (nBS-1) x 2
    %    A(i,:) = [2*(x_i - xRef), 2*(y_i - yRef)]
    A = zeros(nBS-1, 2);
    for b = 2:nBS
        xi = anchorPositions2D(1,b);
        yi = anchorPositions2D(2,b);
        A(b-1,:) = [2*(xi - xRef), 2*(yi - yRef)];
    end

    %% 3) Precompute anchor squares: anchorDistSq(b) = x_b^2 + y_b^2
    anchorDistSq = sum(anchorPositions2D.^2, 1);

    %% 4) Construct B for all points => (nBS-1) x nPoints
    %    B(i,b) = (rRef^2 - rb^2) - [ (xRef^2 + yRef^2) - (x_b^2 + y_b^2) ]
    B = zeros(nBS-1, nPoints);
    for i = 1:nPoints
        rRef   = rangeMatrix(i,1);
        rRefSq = rRef^2;
        for b = 2:nBS
            rb   = rangeMatrix(i,b);
            rbSq = rb^2;
            B(b-1,i) = (rRefSq - rbSq) - ( (xRef^2 + yRef^2) - anchorDistSq(b) );
        end
    end

    %% 5) Solve the least-squares system for all points
    % Instead of A_pinv * B, we use the more stable left division A \ B
    % This yields a (2 x nPoints) result (transposed from the usual perspective).
    X_est = A \ B;
    estPosLS = X_est;  % Each column is [x_est; y_est] for one point

    %% 6) Compute RMSE if we have true positions
    if ~isempty(tgtposMatrix)
        dx = estPosLS(1,:) - tgtposMatrix(1,:);
        dy = estPosLS(2,:) - tgtposMatrix(2,:);
        rmseVec = sqrt(dx.^2 + dy.^2);
    else
        rmseVec = [];
    end

    %% 7) Optional: Intersection areas for each anchor pair, each point
    intersection_areas_cell = cell(nPoints,1);
    for i = 1:nPoints
        % For point i, we have nBS circles => each anchorPositions2D(:,b) is center,
        % and rangeMatrix(i,b) is the radius.
        % We'll store the pairwise intersection areas in a matrix intArea(nBS x nBS).
        intArea = zeros(nBS);
        for p = 1:nBS
            for q = p+1:nBS
                r1 = rangeMatrix(i,p);
                r2 = rangeMatrix(i,q);
                d  = norm(anchorPositions2D(:,p) - anchorPositions2D(:,q));
                intArea(p,q) = calculateIntersectionArea(r1, r2, d);
            end
        end
        intersection_areas_cell{i} = intArea;
    end
end

%% --- Helper function for circle intersection area ---
function A = calculateIntersectionArea(r1, r2, d)
% calculateIntersectionArea Computes the area of intersection between two circles.
%
%   Inputs:
%     r1, r2 : Radii of the two circles
%     d      : Distance between their centers
%
%   Output:
%     A : The area of overlap (0 if no overlap, or if one circle is within another).

    % 1) No intersection case
    if d >= (r1 + r2)
        A = 0;
        return;
    end
    
    % 2) One circle fully inside the other
    if d <= abs(r1 - r2)
        A = pi * min(r1, r2)^2;
        return;
    end

    % 3) Partial overlap
    part1 = r1^2 * acos((d^2 + r1^2 - r2^2) / (2 * d * r1));
    part2 = r2^2 * acos((d^2 + r2^2 - r1^2) / (2 * d * r2));
    part3 = 0.5 * sqrt( (-d + r1 + r2) * ( d + r1 - r2 ) * ...
                        ( d - r1 + r2 ) * ( d + r1 + r2 ) );
    A = part1 + part2 - part3;
end
