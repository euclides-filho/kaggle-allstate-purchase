INSERT INTO TRAIN_TRUNC (
SELECT 
T0.customer_ID,T0.shopping_pt,T0.record_type,T0.day,T0.time,T0.state,T0.location,T0.group_size,T0.homeowner,T0.car_age,T0.car_value,T0.risk_factor,T0.age_oldest,T0.age_youngest,T0.married_couple,T0.C_previous,T0.duration_previous,T0.A,T0.B,T0.C,T0.D,T0.E,T0.F,T0.G,T0.cost
		,T1.customer_ID y_customer_ID,T1.shopping_pt y_shopping_pt,T1.record_type y_record_type,T1.day y_day,T1.time y_time,T1.state y_state,T1.location y_location,T1.group_size y_group_size,T1.homeowner y_homeowner,T1.car_age y_car_age,T1.car_value y_car_value,T1.risk_factor y_risk_factor,T1.age_oldest y_age_oldest,T1.age_youngest y_age_youngest,T1.married_couple y_married_couple,T1.C_previous y_C_previous,T1.duration_previous y_duration_previous,T1.A y_A,T1.B y_B,T1.C y_C,T1.D y_D,T1.E y_E,T1.F y_F,T1.G y_G,T1.cost y_cost
, %s SAMPLE_NUM		
FROM TRAIN T0, (SELECT TI.* , RAND() R FROM TRAIN TI) T1
WHERE T1.RECORD_TYPE = 1 
AND T0.SHOPPING_PT <= IF (T1.R*T1.SHOPPING_PT + IF(T0.SHOPPING_PT <= 5,T1.R/(T0.SHOPPING_PT-2.685),-T1.R/(SQRT(T0.SHOPPING_PT))) > 2, T1.R*T1.SHOPPING_PT + IF(T0.SHOPPING_PT <= 5,T1.R/(T0.SHOPPING_PT-2.685),-T1.R/(SQRT(T0.SHOPPING_PT))), 2)
AND T0.RECORD_TYPE = 0 
AND T1.CUSTOMER_ID = T0.CUSTOMER_ID 

)