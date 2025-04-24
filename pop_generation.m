function waypoint = pop_generation (lat_max, lat_min, long_max, long_min, num_coords, sf_lat, sf_long, pop_size, t_angle, dist_diff_max, dist_diff_min, white_regions)

% % Function test
% sf_lat = 6.411;
% sf_long = 134.656;
% num_coords = 10;
% pop_size = 4;
% lat_max = 7.24;
% lat_min = 5;
% long_max = 136;
% long_min = 132;
% t_angle = 60; % Set the threshold for angle between the pairs
% dist_diff_max = 25*1000; % Max Distance between waypoints (in meters) (50 km)
% dist_diff_min = 20*1000; % Min Distance between waypoints (in meters) (45 km)
% waypoint = zeros(pop_size, num_coords,2);

for k = 1: pop_size
    path_lat = [sf_lat];
    path_long = [sf_long];

    for i = 1:(num_coords-1)
        lat_gen = lat_min + (lat_max - lat_min)*rand;
        long_gen = long_min + (long_max - long_min)*rand;
        angle_between = angle (path_lat(i), path_long(i), lat_gen, long_gen);
        dist_bet = haversine (path_lat(i), path_long(i), lat_gen, long_gen);

        % Check generated latitude and longitude
        while ~(angle_between >= t_angle && dist_bet >= dist_diff_min && dist_bet <= dist_diff_max) || ...
               is_in_white_region(lat_gen, long_gen, white_regions)
            lat_gen = lat_min + (lat_max - lat_min)*rand;
            long_gen = long_min + (long_max - long_min)*rand;
            angle_between = angle (path_lat(i), path_long(i), lat_gen, long_gen);
            dist_bet = haversine (path_lat(i), path_long(i), lat_gen, long_gen);
        end 
    path_lat =[path_lat, lat_gen];
    path_long = [path_long, long_gen];       
    end
waypoint(k, :, 1) = path_lat;
waypoint(k, :, 2) = path_long;
end
end

%% Function for checking angle between coordinate pairs

function bearing = angle(lat1, lon1, lat2, lon2)
    % Function to calculate the compass bearing from point B (lat1, lon1)
    % to point A (lat2, lon2)
    % Inputs:
    %   lat1, lon1 - Latitude and Longitude of starting point (degrees)
    %   lat2, lon2 - Latitude and Longitude of destination point (degrees)
    % Output:
    %   bearing - Compass heading in degrees (0 to 360)
    
    % Convert degrees to radians
    lat1 = deg2rad(lat1);
    lon1 = deg2rad(lon1);
    lat2 = deg2rad(lat2);
    lon2 = deg2rad(lon2);
    
    % Compute the difference in longitude
    delta_lon = lon2 - lon1;
    
    % Compute bearing using the formula
    x = sin(delta_lon) * cos(lat2);
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon);
    
    % Calculate initial bearing in radians
    initial_bearing = atan2(x, y);
    
    % Convert from radians to degrees
    initial_bearing = rad2deg(initial_bearing);
    
    % Normalize to compass bearing (0 to 360 degrees)
    bearing = mod(initial_bearing + 360, 360);
end


function d = haversine(lat1, lon1, lat2, lon2)
    R = 6371000; % Earth's radius in meters
    dlat = deg2rad(lat2 - lat1);
    dlon = deg2rad(lon2 - lon1);
    a = sin(dlat/2) * sin(dlat/2) + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2) * sin(dlon/2);
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    d = R * c;
end

%% Function for checking if a point is in any land region
function in_region = is_in_white_region(lat, lon, white_regions)
    in_region = false;
    for j = 1:size(white_regions, 1)
        lat_min = white_regions(j, 1);
        lat_max = white_regions(j, 2);
        lon_min = white_regions(j, 3);
        lon_max = white_regions(j, 4);
        
        if (lat >= lat_min && lat <= lat_max) && (lon >= lon_min && lon <= lon_max)
            in_region = true;
            return;
        end
    end
end