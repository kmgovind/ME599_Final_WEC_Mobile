function [mission_time, total_desalination_2, mission_length,  f_max_lat, f_max_long, path_lat, path_long, distance_fmax] = waypoint_seq (dir_lat, dir_long, white_regions)

close all;

%% Waypoint Sequence Simulation 
data = readtable('Palau_interpolated_b7000.xlsx');
latitude = round(data.lat, 9);  
longitude = round(data.long, 9);
velocity = round(data.Interpolated_TS, 9); 
SWH = round(data.SWH1, 9);
MWP = round(data.MWP1,9);
desalination = round(data.Interpolated_Power, 9);

% Define boat paremeters
tank_size = 100; % in Liters

% Conversion Factor from Watts to L/hr
conv = 0.055; % (L/hr)/Watts

%% Part where s_max is figured out

% Array to store boat's path
path_lat = [dir_lat'];
path_long = [dir_long'];

% Finer points with 0.01 increments between the waypoints
fine_path_lat = [];
fine_path_long = [];

for i = 1:(length(path_lat)-1)
    % Number of steps needed based on 0.01 increment
    lat_diff = path_lat(i+1) - path_lat(i);
    long_diff = path_long(i+1) - path_long(i);
    
    num_steps = max(ceil(abs(lat_diff) / 0.01), ceil(abs(long_diff) / 0.01));
    
    % Finer points between the current and next point
    lat_steps = linspace(path_lat(i), path_lat(i+1), num_steps);
    long_steps = linspace(path_long(i), path_long(i+1), num_steps);
    
    % Adding these points to the finer path arrays without the last point
    fine_path_lat = [fine_path_lat, lat_steps(1:end-1)];  
    fine_path_long = [fine_path_long, long_steps(1:end-1)];
end

% Add the last point to complete the path
fine_path_lat = [fine_path_lat, path_lat(end)];
fine_path_long = [fine_path_long, path_long(end)];

% Update the arrays with the finer points
path_lat = fine_path_lat;
path_long = fine_path_long;

%% Interpolating the velocity over the grid ScatteredInterpolant
F_velocity = scatteredInterpolant(latitude, longitude, velocity, 'natural', 'linear');
F_SWH = scatteredInterpolant(latitude, longitude, SWH, 'natural', 'linear');
F_MWP = scatteredInterpolant(latitude, longitude, MWP, 'natural', 'nearest');
F_Desal = scatteredInterpolant(latitude, longitude, desalination, 'natural', 'nearest');

% Initialize Arrays
time_series = [];
desalination_series = [];
distance_series = [];
SWH_series = [];
MWP_series = [];
index_series =[];
desal_rate = [];
segment_distances = [];
segment_times = [];
segment_velocities = [];
velocity_series = [];

% Creating the grid initially
grid_lat = min(latitude):0.083:max(latitude);  % Creating a latitude grid
grid_long = min(longitude):0.083:max(longitude);  % Creating a longitude grid

% Initialize time-related variables
simulation_time = 0;
time_step = 10800; % 3 hrs in secs
current_time_index = 1;
total_desalination = 0;
total_distance = 0;
total_time = 0;

% Simulate boat movement along path
for i = 1:(length(path_lat) - 1)
    % Calculate the distance to cover in the segment (meters) using
    % haversine formula
    segment_distance = haversine(path_lat(i), path_long(i), path_lat(i+1), path_long(i+1));
    
    % Use ScatteredInterpolant to find the velocity at the starting point of the segment
    swh_start = F_SWH(path_lat(i), path_long(i));
    mwp_start = F_MWP(path_lat(i), path_long(i));
    v_start = F_velocity(path_lat(i), path_long(i));
    % Handle NaN velocity values
    if isnan(v_start)
        v_start = 0;  % Assign mean value if NaN found
    end
    
    % Interpolate the desalination at the starting point of the segment
    desalination_value = F_Desal(path_lat(i), path_long(i)) * conv;

    % Handle NaN desalination values
    if isnan(desalination_value)
        desalination_value = 0 * conv;  % Assign a mean value 
    end
            
    % Time for the segment (in secs)
    segment_time = segment_distance / v_start;
    
    % Update total distance and time
    total_distance = total_distance + segment_distance;
    total_time = total_time + segment_time;
    simulation_time = simulation_time + segment_time;
    total_desalination = total_desalination + desalination_value * segment_time/3600;

    % Store segment data
    segment_distances = [segment_distances; segment_distance];
    segment_times = [segment_times; segment_time];
    segment_velocities = [segment_velocities; v_start];
    velocity_series = [velocity_series; v_start];
    desalination_series = [desalination_series; total_desalination];
    desal_rate = [desal_rate; desalination_value];
    distance_series = [distance_series; total_distance];
    time_series = [time_series; total_time];
    index_series = [index_series; i];
    SWH_series = [SWH_series; swh_start];
    MWP_series = [MWP_series; mwp_start];

    if desalination_series(i) >= tank_size/2
        i_smax = i;
        break;
    end

end
   
%% Haversine function to calculate distance between two lat/long pairs
function d = haversine(lat1, lon1, lat2, lon2)
    R = 6371000; % Earth's radius in meters
    dlat = deg2rad(lat2 - lat1);
    dlon = deg2rad(lon2 - lon1);
    a = sin(dlat/2) * sin(dlat/2) + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2) * sin(dlon/2);
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    d = R * c;
end

%% Part where boat goes till f_max, stays there and traces way back
[f_max, i_fmax] = max(desal_rate);
f_max = f_max;
distance_fmax = distance_series(i_fmax);
f_max_lat = path_lat(i_fmax);
f_max_long = path_long(i_fmax);

%% Regenerating path from initial point to f_max

% Cutting path vector till f_max
path_lat = [path_lat(1:i_fmax)];
path_long = [path_long(1:i_fmax)];
logic = 1 ; % Boat going out

%% If the starting point has the maximum desalination rate
if i_fmax == 1
    swh_start = F_SWH(f_max_lat, f_max_long);
    mwp_start = F_MWP(f_max_lat, f_max_long);
    desalination_value = F_Desal(f_max_lat, f_max_long)*conv;
    mission_time = tank_size/desalination_value; 
    mission_length = 0; 
    total_desalination_2 = tank_size;
    skip_return_journey = true; % Trigger to stop execution of the rest of the code
else
    skip_return_journey = false;

end

% If skip_return_journey is true, exit the function early before running `time_series_2`
if skip_return_journey
    return; % This will prevent execution of time_series_2 and beyond
end

%% Initialize Arrays for f_max mission
time_series_2 = [];
desalination_series_2 = [];
distance_series_2 = [];
SWH_series_2 = [];
MWP_series_2 = [];
index_series_2 =[];
desal_rate_2 = [];
segment_distances_2 = [];
segment_times_2 = [];
segment_velocities_2 = [];
velocity_series_2 = [];
path_lat_2 = [];

% Initialize variables for f_max mission
simulation_time_2 = 0;
total_desalination_2 = 0;
total_distance_2 = 0;
total_time_2 = 0;
k = 1; % Logic changing counter

% Simulate boat movement to f_max

while true
switch logic
    case 1 % Boat moves at maximum speed
    for i = 1:(length(path_lat) - 1)
        % Switch to stationary desalination at location of f_max
        % Calculate the distance to cover in the segment (meters) using
        % haversine formula
        segment_distance_2 = haversine(path_lat(i), path_long(i), path_lat(i+1), path_long(i+1));
        
        % Use ScatteredInterpolant to find the velocity at the starting point of the segment
        swh_start = F_SWH(path_lat(i), path_long(i));
        mwp_start = F_MWP(path_lat(i), path_long(i));
        v_start = F_velocity(path_lat(i), path_long(i));
        % Handle NaN velocity values
        if isnan(v_start)
            v_start = 0;  % Assign mean value if NaN found
        end
        
        % Interpolate the desalination at the starting point of the segment
        desalination_value = F_Desal(path_lat(i), path_long(i))*conv;
    
        % Handle NaN desalination values
        if isnan(desalination_value)
            desalination_value = 0*conv;  % Assign a mean value 
        end
                
        % Time for the segment (in secs)
        segment_time_2 = segment_distance_2 / v_start;
        
        % Update total distance and time
        total_distance_2 = total_distance_2 + segment_distance_2;
        total_time_2 = total_time_2 + segment_time_2;
        simulation_time_2 = simulation_time_2 + segment_time_2;
        total_desalination_2 = total_desalination_2 + desalination_value * segment_time_2/3600;

        % Store segment data
        segment_distances_2 = [segment_distances_2; segment_distance_2];
        segment_times_2 = [segment_times_2; segment_time_2];
        segment_velocities_2 = [segment_velocities_2; v_start];
        velocity_series_2 = [velocity_series_2; v_start];
        desalination_series_2 = [desalination_series_2; total_desalination_2];
        desal_rate_2 = [desal_rate_2; desalination_value];
        distance_series_2 = [distance_series_2; total_distance_2];
        time_series_2 = [time_series_2; total_time_2];
        index_series_2 = [index_series_2; i];
        SWH_series_2 = [SWH_series_2; swh_start];
        MWP_series_2 = [MWP_series_2; mwp_start];
        path_lat_2 = [path_lat_2; path_lat(i)];
                    
 
        k = k+1;
        if k == i_fmax
            logic = 2;
            break;
        end
    end
   
    case 2 % Desalinate sitting at a position: no movement
        tank_empty_half = (tank_size*0.5) - desalination_series_2(i);
        desalination_value = F_Desal(path_lat(i), path_long(i))*conv;
        desal_rate_2 = [desal_rate_2; desalination_value];
        SWH_series_2 = [SWH_series_2; swh_start];
        MWP_series_2 = [MWP_series_2; mwp_start];
        Req_time = (tank_empty_half/desalination_value)*3600; % Convert to seconds
        segment_times_2 = [segment_times_2; Req_time*2];
        total_time_2 = total_time_2 + Req_time*2;
        time_series_2 = [time_series_2; total_time_2];
        total_desalination_2 = total_desalination_2 + (desalination_value*2 * Req_time/3600);
        desalination_series_2 = [desalination_series_2; total_desalination_2];
        velocity_series_2 = [velocity_series_2; 0];
        distance_series_2 = [distance_series_2;distance_series_2(i)];
        path_lat_2 = [path_lat_2; path_lat(i)];
        logic = 3;
        continue;
      
    case 3 % Tank filled Half. Return to base
    
    % % Cutting path lat and path long to current i
    % path_lat = path_lat(1:i);
    % path_long = path_long(1:i);

    for i = (length(path_lat)-1):-1:1
    % Boat moves back along the same path
        if i > 1
            segment_distance_2 = haversine(path_lat(i-1), path_long(i-1), path_lat(i), path_long(i));
            swh_start = F_SWH(path_lat(i-1), path_long(i-1));
            mwp_start = F_MWP(path_lat(i-1), path_long(i-1));
        elseif i == 1
            segment_distance_2 = segment_distances_2(1);
            swh_start = F_SWH(path_lat(i), path_long(i));
            mwp_start = F_MWP(path_lat(i), path_long(i));
            k= k-1;
        end
    v_start = F_velocity(path_lat(i), path_long(i));
    desalination_value = F_Desal(path_lat(i), path_long(i))*conv;
    if isnan(v_start)
        v_start = 0;  % Assign mean value if NaN found
    end
    if isnan(desalination_value)
        desalination_value = 0;  % Assign mean value if NaN found
    end

    % Time for the segment (in secs)
    segment_time_2 = segment_distance_2 / v_start;
    
    % Update total distance and time
    total_distance_2 = total_distance_2 + segment_distance_2;
    total_time_2 = total_time_2 + segment_time_2;
    simulation_time_2 = simulation_time_2 + segment_time_2;
    total_desalination_2 = total_desalination_2 + desalination_value * segment_time_2/3600;

    % Store segment data
    segment_distances_2 = [segment_distances_2; segment_distance_2];
    segment_times_2 = [segment_times_2; segment_time_2];
    segment_velocities_2 = [segment_velocities_2; v_start];
    velocity_series_2 = [velocity_series_2; v_start];
    desalination_series_2 = [desalination_series_2; total_desalination_2];
    desal_rate_2 = [desal_rate_2; desalination_value];
    distance_series_2 = [distance_series_2; total_distance_2];
    time_series_2 = [time_series_2; total_time_2];
    index_series_2 = [index_series_2; i];
    SWH_series_2 = [SWH_series_2; swh_start];
    MWP_series_2 = [MWP_series_2; mwp_start];
    path_lat_2 = [path_lat_2; path_lat(i)];

    k = k-1;
    if k == 0
        break;
    end

    end
end
if k == 0;
    disp('Mission Complete!');
    break;
end
end

% Convert to hours and kms
mission_time = total_time_2/3600;
mission_length = total_distance_2/1000;
avg_speed = total_distance_2/total_time_2;
avg_desalrate = total_desalination_2/mission_time;

end