-- Find country by IP
CREATE OR REPLACE FUNCTION find_country(ip_int BIGINT) 
RETURNS TEXT AS $$
DECLARE
    lower_id INT := (SELECT MIN(id) FROM ip_data);
    upper_id INT := (SELECT MAX(id) FROM ip_data);
    mid_id INT;
    result RECORD;
    counter INT := 0;
BEGIN
    WHILE lower_id <= upper_id AND counter < 30 LOOP
        mid_id := (lower_id + upper_id) / 2;

        SELECT id, start_ip_int, end_ip_int, country
        INTO result
        FROM ip_data
        WHERE id = mid_id;

        IF result.start_ip_int <= ip_int AND result.end_ip_int >= ip_int THEN
            RETURN result.country;  -- Found the correct range
        ELSIF result.start_ip_int > ip_int THEN
            upper_id := mid_id - 1;  -- Search in the lower half
        ELSE
            lower_id := mid_id + 1;  -- Search in the upper half
        END IF;

        counter := counter + 1;
    END LOOP;

    RETURN 'Unknown';  -- If not found
END;
$$ LANGUAGE plpgsql;


-- Find location (country, latitude, longitude) by IP
CREATE OR REPLACE FUNCTION find_location(ip_int BIGINT) 
RETURNS TABLE (country TEXT, latitude FLOAT, longitude FLOAT) AS $$
DECLARE
    lower_id INT := (SELECT MIN(id) FROM ip_data);
    upper_id INT := (SELECT MAX(id) FROM ip_data);
    mid_id INT;
    result RECORD;
    counter INT := 0;
BEGIN
    WHILE lower_id <= upper_id AND counter < 30 LOOP
        mid_id := (lower_id + upper_id) / 2;

        SELECT id, start_ip_int, end_ip_int, ip_data.country, ip_data.latitude, ip_data.longitude
        INTO result
        FROM ip_data
        WHERE id = mid_id;

        IF result.start_ip_int <= ip_int AND result.end_ip_int >= ip_int THEN
            RETURN QUERY SELECT result.country, result.latitude, result.longitude limit 1;  
			RETURN;
        ELSIF result.start_ip_int > ip_int THEN
            upper_id := mid_id - 1;  
        ELSE
            lower_id := mid_id + 1;  
        END IF;

        counter := counter + 1;
    END LOOP;

    -- If no match is found, return NULL values
    RETURN QUERY SELECT 'Unknown', NULL::FLOAT, NULL::FLOAT;
END;
$$ LANGUAGE plpgsql;


-- Find anycast by IP
CREATE OR REPLACE FUNCTION find_anycast(ip_int BIGINT) 
RETURNS TEXT AS $$
DECLARE
    lower_id INT := (SELECT MIN(id) FROM anycast_data);
    upper_id INT := (SELECT MAX(id) FROM anycast_data);
    mid_id INT;
    result RECORD;
    counter INT := 0;
BEGIN
    WHILE lower_id <= upper_id AND counter < 30 LOOP
        mid_id := (lower_id + upper_id) / 2;

        SELECT id, start_ip_int, end_ip_int
        INTO result
        FROM anycast_data
        WHERE id = mid_id;

        IF result.start_ip_int <= ip_int AND result.end_ip_int >= ip_int THEN
            RETURN true;  -- Found the correct range
        ELSIF result.start_ip_int > ip_int THEN
            upper_id := mid_id - 1;  -- Search in the lower half
        ELSE
            lower_id := mid_id + 1;  -- Search in the upper half
        END IF;

        counter := counter + 1;
    END LOOP;

    RETURN false;  -- If not found
END;
$$ LANGUAGE plpgsql;
