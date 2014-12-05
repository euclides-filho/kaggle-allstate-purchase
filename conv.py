
states = {
'FL':36,
'NY':35,
'PA':34,
'OH':33,
'MD':32,
'IN':31,
'WA':30,
'CO':29,
'AL':28,
'CT':27,
'TN':26,
'KY':25,
'NV':24,
'MO':23,
'OR':22,
'UT':21,
'OK':20,
'MS':19,
'AR':18,
'WI':17,
'GA':16,
'NH':15,
'ME':14,
'NM':13,
'ID':12,
'RI':11,
'KS':10,
'WV':9,
'IA':8,
'DE':7,
'DC':6,
'MT':5,
'NE':4,
'ND':3,
'WY':2,
'SD':1
    }

car_values = {
'e':9,
'f':8,
'd':7,
'g':6,
'h':5,
'c':4,
'i':3,
'NA':0,
'':0,
'b':2,
'a':1
    }

def conv_time(time):
    time = int(time.replace(':',''))
    if time < 600:
        time = 0
    elif time >= 600:
        time = 1
    elif time >= 1200:
        time = 2
    elif time >= 1800:
        time = 3
    return time 
        
    
def conv_duration_previous(duration_previous):
    if duration_previous == "NA":
        return 0
    else:
        return duration_previous
    
def conv_C_previous(C_previous):
    if C_previous == "NA":
        return 0
    else:
        return C_previous

def conv_state(state):
    return states[state]

def conv_car_value(car_value):
    return car_values[car_value]
