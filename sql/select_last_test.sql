SELECT FDATA.* FROM 
	( 
		SELECT customer_ID, MAX(shopping_pt) shopping_pt FROM TEST TT GROUP BY customer_ID
	) MAX_V,
	(
		SELECT 
		T0.customer_ID,T0.shopping_pt,T0.record_type,T0.day,T0.time,T0.state,T0.location,T0.group_size,T0.homeowner,T0.car_age,T0.car_value,T0.risk_factor,T0.age_oldest,T0.age_youngest,T0.married_couple,T0.C_previous,T0.duration_previous,T0.A,T0.B,T0.C,T0.D,T0.E,T0.F,T0.G,T0.cost
		FROM TEST T0
		WHERE T0.RECORD_TYPE = 0 		
	) FDATA
WHERE MAX_V.customer_ID = FDATA.customer_ID
AND FDATA.SHOPPING_PT = MAX_V.SHOPPING_PT