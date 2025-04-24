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