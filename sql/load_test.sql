

LOAD DATA INFILE  './data/test_v2_P.csv'
    IGNORE
    INTO TABLE ALL_STATE.TEST
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
	IGNORE 1 LINES
	(@dummy,@customer_ID,@shopping_pt,@record_type,@day,@time,@state,@location,@group_size,@homeowner,@car_age,@car_value,@risk_factor,@age_oldest,@age_youngest,@married_couple,@C_previous,@duration_previous,@A,@B,@C,@D,@E,@F,@G,@cost)
	SET	customer_ID = IF(@customer_ID='NA',NULL,@customer_ID)	,
	shopping_pt = IF(@shopping_pt='NA',NULL,@shopping_pt)	,
	record_type = IF(@record_type='NA',NULL,@record_type)	,
	day = IF(@day='NA',NULL,@day)	,
	time = IF(@time='NA',NULL,@time)	,
	state = IF(@state='NA',NULL,@state)	,
	location = IF(@location='NA',NULL,@location)	,
	group_size = IF(@group_size='NA',NULL,@group_size)	,
	homeowner = IF(@homeowner='NA',NULL,@homeowner)	,
	car_age = IF(@car_age='NA',NULL,@car_age)	,
	car_value = IF(@car_value='NA',NULL,@car_value)	,
	risk_factor = IF(@risk_factor='NA',NULL,@risk_factor)	,
	age_oldest = IF(@age_oldest='NA',NULL,@age_oldest)	,
	age_youngest = IF(@age_youngest='NA',NULL,@age_youngest)	,
	married_couple = IF(@married_couple='NA',NULL,@married_couple)	,
	C_previous = IF(@C_previous='NA',NULL,@C_previous)	,
	duration_previous = IF(@duration_previous='NA',NULL,@duration_previous)	,
	A = IF(@A='NA',NULL,@A)	,
	B = IF(@B='NA',NULL,@B)	,
	C = IF(@C='NA',NULL,@C)	,
	D = IF(@D='NA',NULL,@D)	,
	E = IF(@E='NA',NULL,@E)	,
	F = IF(@F='NA',NULL,@F)	,
	G = IF(@G='NA',NULL,@G)	,
	cost = IF(@cost='NA',NULL,@cost)
