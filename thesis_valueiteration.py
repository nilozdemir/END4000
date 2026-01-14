import numpy as np

np.random.seed(42)
#mission
Q = np.array([[0, 0.3, 0.7],
     [0.2, 0, 0.8],
     [0.5, 0.5, 0]])

mus = np.array([8, 1, 4])
#deterioration
P_0 = np.array([[0, 0.10, 0.20, 0.20, 0.30, 0.10, 0.10],
[0, 0, 0.10, 0.15, 0.20, 0.25, 0.30],
[0, 0, 0, 0.20, 0.23, 0.22, 0.35],
[0, 0, 0, 0, 0.30, 0.30, 0.40],
[0, 0, 0, 0, 0, 0.50, 0.50],
[0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 1]])

P_1 = np.array([[0, 0.10, 0.20, 0.20, 0.20, 0.10, 0.20],
[0, 0, 0.05, 0.13, 0.22, 0.25, 0.35],
[0, 0, 0, 0.17, 0.22, 0.23, 0.38],
[0, 0, 0, 0, 0.24, 0.32, 0.44],
[0, 0, 0, 0, 0, 0.40, 0.60],
[0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 1]])

P_2 = np.array([[0, 0.10, 0.10, 0.10, 0.15, 0.20, 0.35],
[0, 0, 0.05, 0.08, 0.22, 0.25, 0.40],
[0, 0, 0, 0.05, 0.24, 0.26, 0.45],
[0, 0, 0, 0, 0.18, 0.32, 0.50],
[0, 0, 0, 0, 0, 0.45, 0.55],
[0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 1]])
P = np.stack([P_0,P_1,P_2])

lambdas_ia = np.array([[4, 5, 5.5, 7.5, 8, 9, 0],
             [2, 3, 5, 6, 6.2, 6.5, 0],
             [4, 5, 6, 7, 8, 10, 0]])

#failure or preventative costs
f_i = [300,50,80]
p_i = [200,10,30]

#i , a = 0, 0
M = 7
#v(i,a) value function
v = np.zeros((3,7))
c = np.zeros((3,7))

def future_cost(i,a):
  coef = (mus[i]+ lambdas_ia[i,a])/(mus[i]+ lambdas_ia[i,a] +0.8)

  qsum = np.dot(Q[i,:], v[:,a])

  q_comp =  ((mus[i])  /(mus[i]+lambdas_ia[i,a]))*qsum

  psum = np.sum(P[i,a,:] *  v[i,:] )         # hadamard and then sum
  #print(psum)
  p_comp =  (lambdas_ia[i,a]  /(mus[i]+lambdas_ia[i,a]))*psum

  gamma_v_i_a = coef*(q_comp + p_comp)
  return gamma_v_i_a


#main loop
decisions = []
theta= 0.001
max_iter = 100
missions, ages = 3, 7
values = []
for iter in range(max_iter):
  delta = 0

  for i in range(missions):
    for a in range(ages):
      if a < M-1:
        dec1 = c[i,a].item() + future_cost(i,a).item()
        dec2 = p_i[i] + c[i,0].item() + future_cost(i,0).item()
        dec = np.argmin((dec1, dec2))
        decisions.append(dec)
        v_new = min((c[i,a].item() + future_cost(i,a).item()) , (p_i[i] + c[i,0].item() + future_cost(i,0).item()))
        values.append(v_new)
        delta = max(delta, np.absolute(v[i,a]- v_new))

        v[i,a] = v_new

      else:
        v_new = f_i[i] + c[i,0].item() +future_cost(i,0).item()
        delta = max(delta, np.absolute(v[i,a]- v_new))
        v[i,a] = v_new
        #a=0
        decisions.append(1)
        values.append(v_new)

  if delta < theta:
    break

print(decisions[-21:])
