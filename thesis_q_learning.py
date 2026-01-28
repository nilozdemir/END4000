import pandas as pd
import numpy as np
#states i_k
Q = np.array([[0, 0.3, 0.7],
     [0.2, 0, 0.8],
     [0.5, 0.5, 0]])

mus = np.array([8, 1, 4])
lambda_bar = 17
Q_new = np.zeros((3,3))
for i in range(3):
  Q_new[i] = Q[i]*mus[i]/lambda_bar
#states a
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

P0_new = np.zeros((7,7))
for a in range(7):
  P0_new[a] = P_0[a]*lambdas_ia[0,a]/lambda_bar

P1_new = np.zeros((7,7))
for a in range(7):
  P1_new[a] = P_1[a]*lambdas_ia[1,a]/lambda_bar

P2_new = np.zeros((7,7))
for a in range(7):
  P2_new[a] = P_2[a]*lambdas_ia[2,a]/lambda_bar

ss = np.empty(7)
for i in range(7):
  ss[i] = 1-(np.sum(Q_new[0])+np.sum(P0_new[i]))
QPS0 = np.empty((7,11))
for i in range(7):
  QPS0[i] = np.concatenate([Q_new[0], P0_new[i], [ss[i].item()]])

ss = np.empty(7)
for i in range(7):
  ss[i] = 1-(np.sum(Q_new[1])+np.sum(P1_new[i]))
QPS1 = np.empty((7,11))
for i in range(7):
  QPS1[i] = np.concatenate([Q_new[1], P1_new[i], [ss[i].item()]])

ss = np.empty(7)
for i in range(7):
  ss[i] = 1-(np.sum(Q_new[2])+np.sum(P2_new[i]))
QPS2 = np.empty((7,11))
for i in range(7):
  QPS2[i] = np.concatenate([Q_new[2], P2_new[i], [ss[i].item()]])

QPS = np.stack([QPS0,QPS1,QPS2])

#Deterministic, uk hariç uniform bir şekilde seçiliyor probabilistic
#En iyi sonucu bununla aldık!
Ik, Ak, Uk = [], [], []
for num in range(50000):
  for ik in range(3):
    for ak in range(7):

      if ak == 0:
        uk = 0

      elif ak == 6:
        uk = 1
      else:
        uk = np.random.choice(2,1).item()

      Ik.append(ik)
      Ak.append(ak)
      Uk.append(uk)

#generating next_states
Jik, Jak, Juk = [], [], []

for num in range(len(Ak)):
  ik, ak, uk = Ik[num], Ak[num], Uk[num]

  if uk == 1:
    ak = 0

  choice = np.random.choice(11,1, p=QPS[ik,ak]).item()
  if choice < 3:
      ik = choice
  elif choice <10:
      ak = choice - 3
  else:
      ik = ik
      ak = ak

  Jak.append(ak)
  Jik.append(ik)

cn_states = pd.DataFrame((Ik, Ak, Uk, Jik, Jak)).T
cn_states.columns = ['ik','ak','uk','nextik','nextak']

#punishments in this case, since it's minimization
reward_0 = np.array([[0, 200],
                   [0, 200],
                   [0, 200],
                   [0, 200],
                   [0, 200],
                   [0, 200],
                   [300, 300]])

reward_1 = np.array([[0, 10],
                    [0, 10],
                    [0, 10],
                    [0, 10],
                    [0, 10],
                    [0, 10],
                    [50, 50]])

reward_2 = np.array([[0, 30],
                     [0, 30],
                     [0, 30],
                     [0, 30],
                     [0, 30],
                     [0, 30],
                     [80, 80]])

rewards = np.stack([reward_0,reward_1,reward_2])
#stepsize gamma
c1 = 5
c2 = 10
steps = 1000000
gammas = []
for k in range(steps):
  gammas.append(c1/(k+c2))

alpha = 0.8
lambda_bar = 17

def check(Q_cube):
  matrix = np.argmin(Q_cube,axis=2)
  matrix[:,6] = 1
  matrix[:,0] = 0
  return matrix

#MAIN LOOP
Q_cube = np.ones((3,7,2))*50
#Q_cube = np.zeros((3,7,2))

for k in range(steps):
  ik, ak, uk, nextik, nextak = cn_states.iloc[k]
  #print(ik,ak,uk)
  for i in range(3):
    for a in range(7):
      for control in range(2):



        if i ==ik and a == ak and control == uk:

          #dec = np.argmin(Q_cube[nextik,nextak]).item()

          reward = rewards[i,a, control]

          discount_factor = lambda_bar/(lambda_bar+alpha)

          if nextak == 6:
            FkQk = reward + discount_factor * Q_cube[nextik,nextak,1]
          elif nextak == 0:
            FkQk = reward + discount_factor * Q_cube[nextik,nextak,0]
          else:
            FkQk = reward + discount_factor * min(Q_cube[nextik,nextak])




          #print(min(Q_cube[nextik,nextak]))
        else:
          FkQk = Q_cube[i,a,control]

        Q_cube[i,a,control] = (1-gammas[k])*Q_cube[i,a,control] + gammas[k]*FkQk

#check results/policies
check(Q_cube)
