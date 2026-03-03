import os
import glob
import json
import csv
import h5py
import numpy as np
from scipy import stats

ALPHA = 0.007297
K_0 = 1.1467e84
NU_PLANCK = 1.8549e43
J_SPIN = 0.5
SQRT_J = np.sqrt(J_SPIN * (J_SPIN + 1.0))

TARGET_APPROX = "C01:IMRPhenomXPHM"


def calculate_octave(m_total, m_final):
   e_loss = m_total - m_final
   valid_idx = (e_loss > 0) & (e_loss < m_total)
   n_star = e_loss[valid_idx] / (m_total[valid_idx] * ALPHA)
   nu = np.sqrt(K_0 / (SQRT_J * n_star))
   n_octave = np.log2(NU_PLANCK / nu)
   return n_octave, valid_idx


def process_h5_datasets(directory="."):
   h5_files = glob.glob(os.path.join(directory, "*.h5")) + glob.glob(os.path.join(directory, "*.hdf5"))

   if not h5_files:
       print("[!] No .h5 or .hdf5 files found in the current directory.")
       return

   print("[*] Found " + str(len(h5_files)) + " HDF5 files. Processing Seraphim geometry...")

   results = []

   for filepath in h5_files:
       filename = os.path.basename(filepath)
       try:
           with h5py.File(filepath, "r") as f:
               if TARGET_APPROX in f:
                   posteriors = f[TARGET_APPROX]["posterior_samples"]
               elif "posterior_samples" in f:
                   posteriors = f["posterior_samples"]
               else:
                   keys = list(f.keys())
                   approx_key = next((k for k in keys if "IMR" in k or "SEOBNR" in k), keys[0])
                   posteriors = f[approx_key]["posterior_samples"]

               m1 = np.array(posteriors["mass_1"])
               m2 = np.array(posteriors["mass_2"])
               m_final = np.array(posteriors["final_mass"])
               chi_eff = np.array(posteriors["chi_eff"])

               m_total = m1 + m2
               q = m2 / m1

               n_array, valid_idx = calculate_octave(m_total, m_final)

               q_valid = q[valid_idx]
               chi_eff_valid = chi_eff[valid_idx]

               if len(n_array) < 100:
                   print("[-] Skipping " + filename + ": Insufficient valid posteriors.")
                   continue

               event_data = {
                   "event_file": filename,
                   "median_n": float(np.median(n_array)),
                   "std_n": float(np.std(n_array)),
                   "median_q": float(np.median(q_valid)),
                   "median_chi_eff": float(np.median(chi_eff_valid)),
                   "valid_samples": len(n_array),
               }
               results.append(event_data)

       except Exception as e:
           print("[!] Error reading " + filename + ": " + str(e))

   if len(results) > 2:
       n_vals = np.array([r["median_n"] for r in results])
       chi_vals = np.array([r["median_chi_eff"] for r in results])
       q_vals = np.array([r["median_q"] for r in results])

       X = np.column_stack((np.ones(len(n_vals)), chi_vals, q_vals))
       beta, residuals, rank, s = np.linalg.lstsq(X, n_vals, rcond=None)

       print("\n=== Multivariate Regression Results ===")
       print("Intercept:        " + str(round(beta[0], 4)))
       print("chi_eff slope:    " + str(round(beta[1], 4)) + "  (Theoretical target: ~0.148)")
       print("mass_ratio slope: " + str(round(beta[2], 4)))

       if abs(beta[1]) > abs(beta[2]):
           print("\n[+] Spin maintains dominance over mass-ratio. The geometry holds.")
       else:
           print("\n[!] Mass-ratio exerts significant confounding leverage. Review phase boundary.")

   json_path = "seraphim_results.json"
   csv_path = "seraphim_results.csv"

   with open(json_path, "w") as jf:
       json.dump(results, jf, indent=4)

   if results:
       with open(csv_path, "w", newline="") as cf:
           writer = csv.DictWriter(cf, fieldnames=results[0].keys())
           writer.writeheader()
           writer.writerows(results)

   print("\n[*] Extracted data saved to '" + json_path + "' and '" + csv_path + "'.")


if __name__ == "__main__":
   process_h5_datasets()
