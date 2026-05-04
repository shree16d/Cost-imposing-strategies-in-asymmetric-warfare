import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DRONES = {
    "FPV Drone":    500,
    "Shahed-136":   35000,
}

Cf = 150

DEFENSE = {
    "Patriot":     3729769,
    "IRIS-T":      450000,
    "Iron Dome":   20000,
    "C_RAM":       8100,
    "Iron Beam":   3.50,
}

def f1_attack_cost(r, Cr, f):
    return (r * Cr) + (f * Cf)

def f2_wasted(I, f, N, Pd=1.0):
    if N == 0: return 0
    return I * (f / N) * (1 - Pd) 

def f3_real_intercepts(I, FI):
    return I - FI

def f4_intercept_rate(RI, r):
    if r == 0: return 0
    return min(1.0, RI  / r)

def f5_through(r, Irate):
    return r * (1 - Irate)

def f6_success_rate(r, RI):
    if r == 0: return 0
    return max(0, (r - RI) / r)

def f7_defense_cost(I, Ci):
    return I * Ci

def f8_cer(Cd, Ca):
    return Cd / Ca if Ca > 0 else 0

def calculate(N, decoy_ratio, drone, defense, Pd=1.0):
    Cr   = DRONES[drone]
    Ci   = DEFENSE[defense]
    f = int(N * decoy_ratio)
    r = N - f
    I  = int(N * 0.5)

    Ca      = f1_attack_cost(r, Cr, f)
    FI      = f2_wasted(I, f, N,  Pd)
    RI      = f3_real_intercepts(I, FI)
    Irate   = f4_intercept_rate(RI, r)
    Through = f5_through(r, Irate)
    ASR      = f6_success_rate(r, RI)
    Cf      = f7_defense_cost(I, Ci)
    CER     = f8_cer(Cf, Ca)

    return{"N":N, "r":r, "f":f, "Ca":Ca, "Cf":Cf, "Pd":Pd,
            "Irate":Irate, "ASR":ASR, "CER":CER,
            "drone":drone, "defense":defense}

import pandas as pd
def show_results():
    print("Attacker Vs Defender(If Swarm size = 50, Decoy drones = 20%)")
    
    all_data = [] 

    for drone in DRONES:
        for defense in DEFENSE:
            result = calculate(50, 0.20, drone, defense)
            if result["CER"] > 1:
                winner = "Attacker"
            else:
                winner = "Defender"
            print(f"Drone: {drone} vs {defense}")
            print(f"   Winner: {winner} (CER: {result['CER']})")
            all_data.append(result)
    df = pd.DataFrame(all_data)
    df.to_csv("results.csv", index=False)
    print("\nSUCCESS")

def show_decoy_effect():
    print("\nHOW DECOYS HELP (Instance : FPV vs Patriot)")
    
    percentages = [0, 0.2, 0.4, 0.6, 0.8]
    for amount in percentages:
        result = calculate(50, amount, "FPV Drone", "Patriot")
        percent = amount * 100
        print(f"With {percent}% decoys, the CER is {result['CER']}")

def show_swarm_effect():
    print("\nHOW SWARM SIZE EFFECT")
    swarm_sizes = [10, 25, 50, 100, 150, 200]
    for size in swarm_sizes:
        patriot_result = calculate(size, 0.20, "FPV Drone", "Patriot")
        irondome_result = calculate(size, 0.20, "FPV Drone", "Iron Dome")
        
        print(f"Swarm of {size}: Patriot CER = {patriot_result['CER']}, Iron Dome CER = {irondome_result['CER']}")

def monte_carlo_detection():
    print("SENSOR NOISE")

    N = 50
    drone = "FPV Drone"
    defense = "Patriot"

    mean = 0.85 # Assumption
    std  = 0.07 # Assumption 
    # formulas of variance and mean is derived from beta distribution 
    var = std**2
    k = mean*(1-mean)/var - 1
    alpha = mean * k
    beta  = (1-mean) * k

    samples = np.random.beta(alpha, beta, 1000)

    cer_list = []
    asr_list = []

    for Pd in samples:
        res = calculate(N, 0.20, drone, defense, Pd)
        cer_list.append(res["CER"])
        asr_list.append(res["ASR"])

    print("CER mean:", np.mean(cer_list))
    print("ASR mean:", np.mean(asr_list))

def optimize_attacker(budget):
    print("\n ATTACKER OPTIMIZATION ")

    best = None
    best_score = -1

    for r in range(1, 60):
        for f in range(0, 80):

            Cr = DRONES["FPV Drone"]
            Ca = f1_attack_cost(r, Cr, f)

            if Ca > budget:
                continue

            N = r + f
            res = calculate(N, f/N if N>0 else 0, "FPV Drone", "Patriot")

            ASR = res["ASR"]

            if ASR < 0.2:
                continue

            score = ASR - 0.00001 * Ca

            if score > best_score:
                best_score = score
                best = res

    print("Best Attack Strategy:", best)

def optimize_defender(budget):
    print("\n DEFENDER OPTIMIZATION ")

    best = None
    best_score = -1

    for I in range(1, 200):

        Ci = DEFENSE["Patriot"]
        Cd = f7_defense_cost(I, Ci)

        if Cd > budget:
            break

        res = calculate(50, 0.20, "FPV Drone", "Patriot", I)

        Irate = res["Irate"]

        if Irate < 0.7:
            continue

        score = Irate - 0.000001 * Cd

        if score > best_score:
            best_score = score
            best = {"I": I, "Irate": Irate, "Cost": Cd}

    print("Best Defense Strategy:", best)

def make_charts():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Asymmetric Drone Warfare — Economic Model", fontsize=13, fontweight="bold")

    swarm = list(range(5, 205, 5))
    colors = {"Patriot":"#E24B4A",
    "Iron Beam":"#1D9E75",
    "IRIS-T":"#378ADD",
    "Iron Dome":"#7F77DD"
}
    for d, c in colors.items():
        cers = [calculate(N, 0.20, "FPV Drone", d)["CER"] for N in swarm]
        axes[0].plot(swarm, cers, color=c, linewidth=2, label=d)
    axes[0].axhline(y=1, color="black", linestyle="--", linewidth=1.5, label="CER=1")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Swarm Size (N)")
    axes[0].set_ylabel("CER (log scale)")
    axes[0].set_title("CER vs Swarm Size\n(FPV Drone, 20% Decoys)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    decoy_pct = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    cers = [calculate(50, dr/100, "FPV Drone", "Patriot")["CER"] for dr in decoy_pct]
    axes[1].plot(decoy_pct, cers, color="#E24B4A", linewidth=2.5,
                 marker="o", markersize=8, markerfacecolor="white", markeredgewidth=2)
    axes[1].set_xlabel("Decoy Ratio (%)")
    axes[1].set_ylabel("CER")
    axes[1].set_title("Decoy Drones Effect\n(N=50, FPV vs Patriot)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("charts1.png", dpi=150, bbox_inches="tight")
    print("\nSaved: charts.png")

show_results()
show_decoy_effect()
show_swarm_effect()
make_charts()
monte_carlo_detection()
optimize_attacker(500000)
optimize_defender(3000000)
