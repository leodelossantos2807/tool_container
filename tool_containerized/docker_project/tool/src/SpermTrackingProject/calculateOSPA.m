function [d_ospa eps_loc eps_card] = calculateOSPA(zTotal, TrackRecord)

OSPA.p = 1;
OSPA.l = 25;
OSPA.c = 50;

% Number of Frames
numFrames = length(unique(zTotal(3,:)));

% Number of True Tracks
m = length(unique(zTotal(4,:)));

% Number of Estimated Tracksplot
n = length(unique(TrackRecord(:,1)));



% /////////////////////////////////////////////////////////////////////// %
%
%   Assign Labels
%
% /////////////////////////////////////////////////////////////////////// %

% Distance Cut-off 
DELTA = 100;

% Initialize the Distance Matrix
D = zeros(m,n);

% For Each True Track
for i = 1:m        

    % True Track Measured Position
    % x = zTotal([1:3], (zTotal(4,:) == i));
    
    % True Track i True Position and time 
    x = zTotal([5 6 3], (zTotal(4,:) == i));
    
    % For Each Estimated Track
    for j = 1:n
        
        % Estimated Track j 
        y = TrackRecord((TrackRecord(:,1) == j), [19:20 4])';
        
        % Measurements
        % y = TrackRecord((TrackRecord(:,1) == j), [5 6 4])';
        
        % For Each Video Frame
        for k = 1:numFrames
            
            % Find the time index in xt and yt corresponding to now
            x_idx = find(x(3,:) == k);
            y_idx = find(y(3,:) == k);
            
            % If both tracks exist at the same time, calculate their dist
            if ~isempty(x_idx) && ~isempty(y_idx)
                d = norm(x(1:2, x_idx) - y(1:2, y_idx));
                D(i,j) = D(i,j) + min(DELTA, d);
            else
                D(i,j) = D(i,j) + DELTA;
            end
        end
        
        % Average distance over all frames
        D(i,j) = D(i,j) / numFrames;
        
    end    
        
end

% Label the estimated and GT tracks
label_matrix = munkres(D);

% Calculate OSPA 
for k = 1:numFrames

    X = [];
    X_label = [];
    
    for i = 1:m
        
        % Ground Truth Track i
        x = zTotal(1:3, (zTotal(4,:) == i));
        
        % Find the time index of x_i at Frame k
        x_idx = find(x(3,:) == k);
    
        % If track x_i exists at time k
        if ~isempty(x_idx)
            X = [X x(1:2, x_idx)];
            X_label = [X_label i];
        end
        
    end
    
    Y = [];
    Y_label = [];
    
    for j = 1:n
        
        % Estimated Track j 
        % y = TrackRecord((TrackRecord(:,1) == j), [19:20 4])';
        y = TrackRecord((TrackRecord(:,1) == j), [5 6 4])';
               
        % Find the time index of y_j at time k
        y_idx = find(y(3,:) == k);
        
        % If track y_j exists at time k
        if ~isempty(y_idx)
            
            Y = [Y y(1:2, y_idx)];
            iy = find(label_matrix(:,j));
            
            if ~isempty(iy)
                Y_label = [Y_label iy];
            else
                Y_label = [Y_label 99999];
            end
            
        end
        
    end
        
       
    % Calculate OSPA-T distance
    [d_ospa(k) eps_loc(k) eps_card(k)] = ...
        trk_ospa_dist(X, X_label, Y, Y_label, OSPA);
          
end








