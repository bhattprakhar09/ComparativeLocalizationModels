function [estPosLS, rmseVec, intersection_areas_cell] = ...
    estimatePositionUsingLS_batchVectorized(anchorPositions2D, rangeMatrix, tgtposMatrix)
% ESTIMATEPOSITIONUSINGLS_BATCH
%
% Vectorized least-squares localization for multiple points. 
%
% Inputs:
%   anchorPositions2D : 2 x nBS    (each column is [x_b; y_b])
%   rangeMatrix       : nPoints x nBS   (row i => ranges for point i)
%   tgtposMatrix      : 2 x nPoints     (optional: if provided, we compute RMSE)
%
% Outputs:
%   estPosLS          : 2 x nPoints     (LS estimates for each point)
%   rmseVec           : 1 x nPoints     (RMSE if tgtposMatrix given; otherwise empty)
%   intersection_areas_cell : cell array of intersection areas if you still want them

    % --- Check inputs ---
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

    % We need at least 3 anchors for 2D LS
    if nBS < 3
        error('At least 3 anchors are required for 2D LS. Found nBS=%d', nBS);
    end

    %% 1) Pick the first anchor as reference
    refPos = anchorPositions2D(:,1);  % [xRef; yRef]
    xRef   = refPos(1);
    yRef   = refPos(2);

    %% 2) Build matrix A => (nBS-1) x 2
    %   A(i-1,:) = [2*(x_i - xRef), 2*(y_i - yRef)]
    % (We use the same approach as your original code but generalized.)
    A = zeros(nBS-1, 2);
    for b = 2:nBS
        xi = anchorPositions2D(1,b);
        yi = anchorPositions2D(2,b);
        A(b-1,:) = [2*(xi - xRef), 2*(yi - yRef)];
    end

    % We will need the “pseudo‐inverse” of A: 
    %   A_pinv = (A' * A)^{-1} * A'
    AtransA   = A.' * A;        % 2x2
    invAtransA= inv(AtransA);   % 2x2
    A_pinv    = invAtransA * A.';  % 2 x (nBS-1)

    %% 3) Precompute anchor squares
    % anchorDistSq(b) = x_b^2 + y_b^2
    anchorDistSq = sum(anchorPositions2D.^2, 1);
    refDistSq    = anchorDistSq(1);

    %% 4) Construct B for *all* points: size => (nBS-1) x nPoints
    B = zeros(nBS-1, nPoints);

    for i = 1:nPoints
        % For the i-th point, the range from the reference anchor:
        rRef = rangeMatrix(i,1);
        rRefSq = rRef^2;
        
        for b = 2:nBS
            rb = rangeMatrix(i,b);
            rbSq = rb^2;
            % b(b-1) = (rRef^2 - rb^2) - (refDistSq - anchorDistSq(b))
            % but note your original code had a sign difference. 
            % In your code: 
            %   b(i-1) = (refRange^2 - ri^2) - (refAnchorPos(1)^2 - xi^2) - (refAnchorPos(2)^2 - yi^2);
            %
            % This expands to:
            %   (rRef^2 - rb^2) - ( (xRef^2 - xi^2) + (yRef^2 - yi^2) )
            %   = (rRef^2 - rb^2) - ( xRef^2 + yRef^2 ) + (xi^2 + yi^2)
            %   = (rRef^2 - rb^2) + (anchorDistSq(b) - refDistSq)
            % 
            % So let's keep that logic consistent:
            B(b-1,i) = (rRefSq - rbSq) - ( (xRef^2 + yRef^2) - anchorDistSq(b) );
        end
    end

    %% 5) Solve for all points:
    %   X_est = A_pinv * B   => shape: (2 x nPoints)
    X_est = A_pinv * B;

    % So X_est(1,i) is the X coordinate, X_est(2,i) is the Y coordinate for the i-th point
    estPosLS = X_est;

    %% 6) Compute RMSE if we have true positions
    if ~isempty(tgtposMatrix)
        dx = estPosLS(1,:) - tgtposMatrix(1,:);
        dy = estPosLS(2,:) - tgtposMatrix(2,:);
        rmseVec = sqrt(dx.^2 + dy.^2);  % 1 x nPoints
    else
        rmseVec = [];
    end

    %% 7) Intersection areas (Optional)
    % If you still want to do pairwise circle intersections for *each* anchor pair,
    % that is typically for a single point. Doing it for *each* point => we get
    % intersection areas for each point. So let's store them in a cell array:
    intersection_areas_cell = cell(nPoints,1);

    for i = 1:nPoints
        % For point i, we have nBS circles => anchorPositions2D + rangeMatrix(i,:)
        % intersection_areas(i, j) for anchors i and j
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

    % Done. The function returns estPosLS, rmseVec, intersection_areas_cell.
end

%% Helper for circle intersection area
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
