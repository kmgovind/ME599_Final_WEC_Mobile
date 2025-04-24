clc;
clear;
tic;

% Define start and finish locations
sf_lat = 7.324;
sf_long = 134.739;

% Define mission domain parameters
lat_max = 8;
lat_min = 5;
long_max = 136;
long_min = 132;
t_angle = 70; % Set the threshold for angle between the pairs
dist_diff_max = 10*1000; % Max Distance between waypoints (in meters) 
dist_diff_min = 1*1000; % Min Distance between waypoints (in meters) 

% Define land mass regions row-wise in format (lat_min, lat_max, lon_min, lon_max)
white_regions = [
    7.324, 7.822, 134.324, 134.822;
];


% Define Genetic Algorithm Parameters
num_coords = 10; % Number of coordinate pairs to generate.
num_generations = 1000; % Number of generations for evolution
pop_size = 30; % Population size for GA
mutation_rate = 0.1; % Probability of mutation

% Store best fitness for each generation
best_fitness_values = zeros(num_generations, 1);

% Storage for all generations' data
all_generations_data = struct();

% Initialize fitness struct with correct fields
fitnessData = struct('Fitness', [], 'Latitude', [], 'Longitude', [], 'Time', [], 'Desal', []);
fitnessData = repmat(fitnessData, pop_size, 1);

% Initial population of waypoints
waypoint = pop_generation(lat_max, lat_min, long_max, long_min, num_coords, sf_lat, sf_long, pop_size, t_angle, dist_diff_max, dist_diff_min, white_regions);

% Run GA for multiple generations
gen = 1;
while gen <= num_generations
    % Evaluate fitness for each waypoint sequence
    for i = 1:pop_size
        dir_lat = waypoint(i,:,1);
        dir_long = waypoint(i,:,2);
        [time, total_desalination_2] = waypoint_seq(dir_lat, dir_long, white_regions);
        fitness_eval = (-0.5*time+0.5*total_desalination_2)/1000;
        
        % Store fitness and corresponding waypoints
        fitnessData(i).Fitness = fitness_eval;
        fitnessData(i).Latitude = dir_lat;
        fitnessData(i).Longitude = dir_long;
        fitnessData(i).Time = time;
        fitnessData(i).Desal = total_desalination_2;
    end

    if gen >= 50
        recent_fitness = best_fitness_values(gen-4:gen);
        if std(recent_fitness) < 1e-6
            disp(['Stopping early at generation ', num2str(gen), ' due to convergence.']);
            best_fitness_values = best_fitness_values(1:gen);
            break;
        end
    end

    % Store all path's data for this generation
    all_generations_data(gen).Generation = gen;
    all_generations_data(gen).FitnessData = fitnessData;

    % Select best solution
    [~, best_idx] = max([fitnessData.Fitness]);
    best_fitness_values(gen) = fitnessData(best_idx).Fitness;

    % Save data 
    save("599Project7_3.mat", "all_generations_data");

    % GA Evolutionary Process
    new_population = zeros(pop_size, num_coords, 2);
    
    for i = 1:pop_size
        % Selection: Tournament Selection
        parent1 = tournament_selection(fitnessData);
        parent2 = tournament_selection(fitnessData);

        % Crossover with fixed first waypoint
        child = crossover(parent1, parent2, sf_lat, sf_long);

        % Mutation while keeping the first waypoint unchanged
        child = mutate(child, lat_max, lat_min, long_max, long_min, mutation_rate, sf_lat, sf_long, white_regions);

        % Assign to new population
        new_population(i,:,:) = permute(child, [3, 2, 1]); % Ensure correct shape
    end
    gen=gen+1;
    waypoint = new_population; % Update population for next generation
end

%% Extract best fitness data accross generations
% Initialize arrays to store results
max_fitness_per_generation = [];
time_taken_per_generation = [];
latitude_per_generation = {};  % Store as a cell array to keep matrix format
longitude_per_generation = {}; % Store as a cell array to keep matrix format

% Loop through each generation
for i = 1:size(all_generations_data, 2)  % Iterate over each generation
    generation_data = all_generations_data(i).FitnessData; % Access FitnessData
    
    % Extract fitness values
    fitness_values = cellfun(@(x) x(1), {generation_data.Fitness});
    
    % Find the index of the max fitness
    [max_fitness, idx] = max(fitness_values);
    
    % Store the max fitness
    max_fitness_per_generation(i) = max_fitness;
    
    % Extract corresponding time, latitude, and longitude
    time_taken_per_generation(i) = generation_data(idx).Time; 
    latitude_per_generation{i} = generation_data(idx).Latitude;  % Store as matrix
    longitude_per_generation{i} = generation_data(idx).Longitude; % Store as matrix
end

%% Get the overall maximum fitness and corresponding details
[overall_max_fitness, gen_idx] = max(max_fitness_per_generation);
overall_time_taken = time_taken_per_generation(gen_idx);
overall_latitude = latitude_per_generation{gen_idx};  % Keep as matrix
overall_longitude = longitude_per_generation{gen_idx}; % Keep as matrix

% Display the results
disp(['Overall Maximum Fitness: ', num2str(overall_max_fitness)]);
disp(['Time Taken: ', num2str(overall_time_taken)]);
disp('Latitude Matrix:');
disp(overall_latitude);
disp('Longitude Matrix:');
disp(overall_longitude);

% Save Best Coordinates Data
% save("bestCoordsData_PalauTrial5.mat", "overall_latitude","overall_longitude","overall_max_fitness","overall_time_taken");

%% Plot fitness evolution
figure;
plot(1:size(all_generations_data, 2), max_fitness_per_generation, '-o', 'LineWidth', 2);
xlabel('Generation', 'Interpreter','latex', 'FontSize',40);
ylabel('Best Fitness', 'Interpreter','latex', 'FontSize',40);
% ylim([1.86, 1.88]);
xlim([1, 50]);
title('Fitness Evolution Over Generations', 'Interpreter','latex', 'FontSize',40);
grid on;
set(gca, 'FontSize', 35, 'Box', 'on');

% Display best fitness
disp(['Best Fitness Achieved: ', num2str(best_fitness_values(end))]);

%% Tournament Selection Function 
function parent = tournament_selection(fitnessData)
    tournament_size = 2;
    indices = randperm(length(fitnessData), tournament_size);
    [~, best_idx] = max([fitnessData(indices).Fitness]);
    parent = [fitnessData(indices(best_idx)).Latitude; fitnessData(indices(best_idx)).Longitude];
end

%% Crossover Function (Preserving First Waypoint)
function child = crossover(parent1, parent2, sf_lat, sf_long)
    mask = rand(size(parent1)) > 0.5;
    child = parent1 .* mask + parent2 .* (~mask);
    
    % Ensure first waypoint remains unchanged
    child(:,1) = [sf_lat; sf_long];
end

%% Mutation Function (Preserving First Waypoint and Avoiding Land Regions)
function mutated = mutate(child, lat_max, lat_min, long_max, long_min, mutation_rate, sf_lat, sf_long, white_regions)
    if rand < mutation_rate
        mutation_idx = randi([2, size(child, 2)]); % Select a random waypoint (not the first one)
        
        % Generate new coordinates that are not in a land region
        valid_point = false;
        while ~valid_point
            new_lat = lat_min + (lat_max - lat_min) * rand;
            new_lon = long_min + (long_max - long_min) * rand;
            
            % Check if the new coordinate is in a white region %%
            if ~is_in_white_region(new_lat, new_lon, white_regions)
                valid_point = true;
                child(1, mutation_idx) = new_lat;
                child(2, mutation_idx) = new_lon;
            end
        end
    end

    % Ensure first waypoint remains unchanged
    child(:,1) = [sf_lat; sf_long];
    mutated = child;
end


toc;