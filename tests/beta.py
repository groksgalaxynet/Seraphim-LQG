import json
import csv
import numpy as np

G = 6.67430e-11
C_SPEED = 299792458.0
M_SUN = 1.98847e30
N_BBH = 5.335

NSBH_EVENTS = {
   "GW191219": {"m2": 1.31, "n_obs": 3.296},
   "GW190917": {"m2": 2.35, "n_obs": 4.660},
   "GW200105": {"m2": 2.02, "n_obs": 4.688},
   "GW200115": {"m2": 1.50, "n_obs": 4.710},
}

N_ITERATIONS = 1000000


def run_monte_carlo():
   print("[*] Initializing Seraphim Monte Carlo EoS Injection (" + str(N_ITERATIONS) + " iterations)...")

   r_samples = np.random.uniform(10000.0, 14000.0, (N_ITERATIONS, len(NSBH_EVENTS)))

   beta_results = np.zeros((N_ITERATIONS, len(NSBH_EVENTS)))
   event_names = list(NSBH_EVENTS.keys())

   for i, event in enumerate(event_names):
       m2_kg = NSBH_EVENTS[event]["m2"] * M_SUN
       n_obs = NSBH_EVENTS[event]["n_obs"]

       compactness = (G * m2_kg) / (r_samples[:, i] * C_SPEED ** 2)

       k_data = 2 ** (N_BBH - n_obs)

       beta_implied = np.log2(k_data) / np.log2(0.5 / compactness)
       beta_results[:, i] = beta_implied

   global_beta = np.mean(beta_results, axis=1)

   beta_mean = np.mean(global_beta)
   beta_median = np.median(global_beta)
   ci_lower = np.percentile(global_beta, 2.5)
   ci_upper = np.percentile(global_beta, 97.5)

   print("\n=== EoS Monte Carlo Results ===")
   print("Mean beta:    " + str(round(beta_mean, 4)))
   print("Median beta:  " + str(round(beta_median, 4)))
   print("95% CI:       [" + str(round(ci_lower, 4)) + ", " + str(round(ci_upper, 4)) + "]")

   print("\n--- Mechanism Probability ---")
   prob_newtonian = np.mean(global_beta > 0.95) * 100
   prob_redshift = np.mean(global_beta < 0.60) * 100
   print("Probability aligning with Newtonian Compactness (beta ~ 1.0): " + str(round(prob_newtonian, 2)) + "%")
   print("Probability aligning with Gravitational Redshift (beta ~ 0.5): " + str(round(prob_redshift, 2)) + "%")

   output_data = {
       "simulation_parameters": {
           "iterations": N_ITERATIONS,
           "radius_range_km": [10, 14],
           "n_bbh_baseline": N_BBH,
       },
       "global_beta_statistics": {
           "mean": float(beta_mean),
           "median": float(beta_median),
           "ci_95_lower": float(ci_lower),
           "ci_95_upper": float(ci_upper),
       },
   }

   json_path = "seraphim_beta_mc_results.json"
   with open(json_path, "w") as jf:
       json.dump(output_data, jf, indent=4)

   csv_path = "seraphim_beta_mc_results.csv"
   with open(csv_path, "w", newline="") as cf:
       writer = csv.writer(cf)
       writer.writerow(["Metric", "Value"])
       writer.writerow(["Mean Beta", str(round(beta_mean, 6))])
       writer.writerow(["Median Beta", str(round(beta_median, 6))])
       writer.writerow(["95% CI Lower", str(round(ci_lower, 6))])
       writer.writerow(["95% CI Upper", str(round(ci_upper, 6))])

   print("\n[*] Execution complete. Packaged to '" + json_path + "' and '" + csv_path + "'.")


if __name__ == "__main__":
   run_monte_carlo()
