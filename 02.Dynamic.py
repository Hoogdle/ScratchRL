# Dynamic method make equation as updated shape and iterate it.

V = {'L1':0.0, 'L2':0.0}

iter = 0
threshold = 0.0001

while True:
    prev_V = V.copy()

    V['L1'] = 0.5*(-1+ 0.9*V['L1']) + 0.5*(1+0.9*V['L2'])
    V['L2'] = 0.5*(0+0.9*V['L1']) + 0.5*(-1+0.9*V['L2'])
    
    delta_L1 = abs(V['L1'] - prev_V['L1'])
    delta_L2 = abs(V['L2'] - prev_V['L2'])

    delta = max(delta_L1, delta_L2)

    if delta < threshold:
        print(f"ITER COUNT : {iter}")
        print(f"V[L1] = {V['L1']}")        
        print(f"V[L2] = {V['L2']}")
        break
    
    iter += 1

    