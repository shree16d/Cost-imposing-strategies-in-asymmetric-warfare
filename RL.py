import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
data = pd.read_csv("data.csv")
interceptors = [
     {"name": "Iron Beam (Laser)", "cost": 3.50, "Pd": 0.85},
    {"name": "Iron Dome (Tamir)", "cost": 20000, "Pd": 0.90},
    {"name": "Iris-T ","cost":  450000, "Pd": 0.93},
    {"name": "C_RAM","cost": 8100, "Pd": 0.88},
    {"name": "Patriot PAC-3","cost":  3729769, "Pd": 0.97},]
real_drone = [ {"name": "Shahed-136","cost": 35000},
    {"name": "FPV Drone","cost": 500},
    ]
decoy_ratio  = [0.1, 0.3, 0.5, 0.7]
swarm_size   = [10, 50, 100]
shots = [5, 10, 20, 40, 80]
SCENARIOS = [ 
    {"name": "S1 - Baseline", "N": 20,  "dr": 0.50, "def_b": 5_000_000,  "att_b": 500_000},
    {"name": "S2 - Large Swarm","N": 100, "dr": 0.50, "def_b": 10_000_000, "att_b": 2_000_000},
    {"name": "S3 - High Decoy","N": 50,  "dr": 0.80, "def_b": 6_000_000,  "att_b": 800_000},
    {"name": "S4 - Low Decoy","N": 50,  "dr": 0.20, "def_b": 4_000_000,  "att_b": 1_500_000},
    {"name": "S5 - Budget Constrained", "N": 30,  "dr": 0.60, "def_b": 1_000_000,  "att_b": 200_000},]
def attack_cost(r, Cr, f, Cf):
    return r * Cr + f * Cf
def defense_cost(I, Ci):
    return I * Ci
def wasted_intercepts(I, f, N, Pd):
    if N == 0:
        return 0.0
    return I * (f / N) * (1 - Pd)
def real_intercepts(I, FI, r):
    RI = max(0.0, I - FI)
    return min(RI, float(r))
def intercept_rate(RI, r):
    return min(1.0, RI / r) if r > 0 else 0.0
def attack_success_rate(r, RI):
    return max(0.0, (r - RI) / r) if r > 0 else 0.0
def cost_exchange_ratio(Cd, Ca):
    return Cd / Ca if Ca > 0 else float('inf')
def run_battle(r, f, Cr, Cf, I, Ci, Pd):
    N     = r + f
    Ca    = attack_cost(r, Cr, f, Cf)
    Cd    = defense_cost(I, Ci)
    FI    = wasted_intercepts(I, f, N, Pd)
    RI    = real_intercepts(I, FI, r)
    Irate = intercept_rate(RI, r)
    ASR   = attack_success_rate(r, RI)
    CER   = cost_exchange_ratio(Cd, Ca)
    return dict(N=N, r=r, f=f, I=I,
                Ca=round(Ca),Cd=round(Cd),
                FI=round(FI, 2),RI=round(RI, 2),
                Irate=round(Irate, 4),
                ASR=round(ASR, 4),
                CER=round(CER, 4))
 
# state space (For both Attacker and Defender)
#base-3 numbering system is used
def get_state_defender(budget_left, total_budget, N, decoy_ratio, Irate):
    if budget_left > total_budget * 0.66:
        b = 0
    elif budget_left > total_budget* 0.33:
        b = 1
    else:
        b = 2
 
    if N <= 20:
        n = 0
    elif N <= 50:
        n = 1
    else:
        n = 2
 
    if decoy_ratio <= 0.3:
        d = 0
    elif decoy_ratio <= 0.6:
        d = 1
    else:
        d = 2
 
    if Irate <= 0.3:
        i = 0
    elif Irate <= 0.7:
        i = 1
    else:
        i = 2
 
    return b * 27 + n * 9 + d * 3 + i
 
def get_state_attacker(budget_left, budget_total, ASR, I):
    if budget_left > budget_total * 0.66:
        b = 0
    elif budget_left > budget_total * 0.33:
        b = 1
    else:
        b = 2
 
    if ASR <= 0.3:
        a = 0
    elif ASR <= 0.6:
        a = 1
    else:
        a = 2
 
    if I <= 20:
        i = 0
    elif I <= 50:
        i = 1
    else:
        i = 2
 
    return b * 9 + a * 3 + i
#Action space for both Defender and attacker
d_actions = []
for i in range(len(interceptors)):
    for s in range(len(shots)):
        d_actions.append((i, s))
 
a_actions = []
for d in range(len(real_drone)):
    for dr in range(len(decoy_ratio)):
        for n in range(len(swarm_size)):
            a_actions.append((d, dr, n))
 
print(f"Defender has {len(d_actions)} possible actions")
print(f"Attacker has {len(a_actions)} possible actions")
# for simulated results Q-learning is used
d_states = 81   # base 3 number system
a_states = 27   
 
Q_defender = np.zeros((d_states, len(d_actions)))
Q_attacker = np.zeros((a_states, len(a_actions)))
# from the Q_table actions and strategies are randomly tested and best one is selected 
def choose_action(Q_table, state):
    if random.random() < 0.1: #0.1 is the randomness
        return random.randint(0, Q_table.shape[1] - 1)
    else:
        return int(np.argmax(Q_table[state]))
# reward function is implemented using Q-learning
def update_q_table(Q_table, state, action, reward, next_state, alpha, gamma):
    best_reward = np.max(Q_table[next_state])
    current = Q_table[state, action]

    # updating in-place
    Q_table[state, action] = current + alpha * (
        reward + gamma * best_reward - current
    )
# run simulation for one battle 
def run_episode(budget_def, budget_att, N, decoy_ratio,
                Cr, Cf, Ci, Pd, alpha, gamma):
    # For defender side
    f = int(N * decoy_ratio)
    r = N - f
    if r <= 0:
        r = 1  
    state_d = get_state_defender(budget_def, budget_def, N, decoy_ratio, 0.0)
    action_d = choose_action(Q_defender, state_d)
    i_idx, s_idx = d_actions[action_d]
    chosen = interceptors[i_idx]
    num_shots = shots[s_idx]
    Ci_def = chosen["cost"]
    eff_Pd = Pd * chosen["Pd"]
    if eff_Pd > 0.99:
        eff_Pd = 0.99
    result = run_battle(r, f, Cr, Cf, num_shots, Ci_def, eff_Pd)

    reward_d = 2.0 * result["Irate"] - (result["Cd"] / budget_def)
    if result["CER"] < 1.0:
        reward_d += 0.3

    new_budget_def = budget_def - result["Cd"]

    next_d = get_state_defender(
        new_budget_def,
        budget_def,
        N,
        decoy_ratio,
        result["Irate"]
    )

    update_q_table(Q_defender, state_d, action_d, reward_d, next_d, alpha, gamma)

    # For attacker side
    state_a = get_state_attacker(budget_att, budget_att, 0.0, num_shots)

    action_a = choose_action(Q_attacker, state_a)

    d_idx, dr_idx, n_idx = a_actions[action_a]

    drone = real_drone[d_idx]
    dr    = decoy_ratio[dr_idx]
    N_att = swarm_size[n_idx]

    f_att = int(N_att * dr)
    r_att = N_att - f_att
    if r_att <= 0:
        r_att = 1

    Ca = attack_cost(r_att, drone["cost"], f_att, Cf)

    result_a = run_battle(r_att, f_att, drone["cost"], Cf, num_shots, Ci, Pd)

    reward_a = 2.0 * result_a["ASR"] - (Ca / budget_att)

    if result_a["CER"] > 3.0:
        reward_a += 0.5

    new_budget_att = budget_att - Ca

    next_a = get_state_attacker(
        new_budget_att,
        budget_att,
        result_a["ASR"],
        num_shots
    )

    update_q_table(Q_attacker, state_a, action_a, reward_a, next_a, alpha, gamma)

    return reward_d, reward_a 
#Training the model
def train_model(episodes=1000):
    alpha = 0.1      # learning rate (might tune later)
    gamma = 0.9      # discount factor

    defender_rewards = []
    attacker_rewards = []

    print("Training The model")
    for ep in range(episodes):
        row = data.sample().iloc[0]
        Cr = float(row["drone_cost"])
        Cf = float(row["decoy_cost"])
        Ci = float(row["defense_cost"])
        Pd = float(row["Pd"])
        N = random.choice(swarm_size)
        dr= random.choice(decoy_ratio)
        I_def = random.choice(shots)

        d_rew, a_rew = run_episode(5000000, 500000, N, dr, Cr, Cf, Ci, Pd,alpha, gamma)
        defender_rewards.append(d_rew)
        attacker_rewards.append(a_rew)

        if (ep + 1) % 200 == 0:
          print(f"Episode {ep+1} | Def = {round(defender_rewards[-1],3)} | Att = {round(attacker_rewards[-1],3)}")

    print("Training finished.")
    print("\nLast 5 Defender rewards:", defender_rewards[-5:])
    print("Last 5 Attacker rewards:", attacker_rewards[-5:])
    return defender_rewards, attacker_rewards 
   
