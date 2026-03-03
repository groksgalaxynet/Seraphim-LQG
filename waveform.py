import os
import glob
import json
import csv
import h5py
import numpy as np

ALPHA = 0.007297
K_0 = 1.1467e84
NU_PLANCK = 1.8549e43
J_SPIN = 0.5
SQRT_J = np.sqrt(J_SPIN * (J_SPIN + 1.0))

APPROX_A = "C01:IMRPhenomXPHM"
APPROX_B = "C01:SEOBNRv4PHM"


def calculate_octave(m_total, m_final):
   e_loss = m_total - m_final
   valid_idx = (e_loss > 0) & (e_loss < m_total)
   n_star = e_loss[valid_idx] / (m_total[valid_idx] * ALPHA)
   nu = np.sqrt(K_0 / (SQRT_J * n_star))
   n_octave = np.log2(NU_PLANCK / nu)
   return n_octave, valid_idx


def compute_stats(n_array):
   if len(n_array) < 100:
       return None
   mean_n = np.mean(n_array)
   std_n = np.std(n_array)
   cv = (std_n / mean_n) * 100 if mean_n != 0 else 0
   return {"mean": float(mean_n), "std": float(std_n), "cv": float(cv), "samples": len(n_array)}


def run_waveform_delta_test(directory="."):
   h5_files = glob.glob(os.path.join(directory, "*.h5")) + glob.glob(os.path.join(directory, "*.hdf5"))

   if not h5_files:
       print("[!] No .h5 or .hdf5 files found.")
       return

   print("[*] Initializing Waveform Approximant Delta Test across " + str(len(h5_files)) + " events...")

   results = []
   event_medians_A = []
   event_medians_B = []

   for filepath in h5_files:
       filename = os.path.basename(filepath)
       event_result = {"event_file": filename}

       try:
           with h5py.File(filepath, "r") as f:
               if APPROX_A in f:
                   post_A = f[APPROX_A]["posterior_samples"]
                   m_total_A = np.array(post_A["mass_1"]) + np.array(post_A["mass_2"])
                   m_final_A = np.array(post_A["final_mass"])
                   n_arr_A, _ = calculate_octave(m_total_A, m_final_A)
                   stats_A = compute_stats(n_arr_A)
                   if stats_A:
                       event_result["IMRPhenom_CV"] = stats_A["cv"]
                       event_result["IMRPhenom_Mean"] = stats_A["mean"]
                       event_medians_A.append(float(np.median(n_arr_A)))

               if APPROX_B in f:
                   post_B = f[APPROX_B]["posterior_samples"]
                   m_total_B = np.array(post_B["mass_1"]) + np.array(post_B["mass_2"])
                   m_final_B = np.array(post_B["final_mass"])
                   n_arr_B, _ = calculate_octave(m_total_B, m_final_B)
                   stats_B = compute_stats(n_arr_B)
                   if stats_B:
                       event_result["SEOBNR_CV"] = stats_B["cv"]
                       event_result["SEOBNR_Mean"] = stats_B["mean"]
                       event_medians_B.append(float(np.median(n_arr_B)))

               if "IMRPhenom_CV" in event_result and "SEOBNR_CV" in event_result:
                   event_result["Delta_Mean"] = abs(event_result["IMRPhenom_Mean"] - event_result["SEOBNR_Mean"])
                   results.append(event_result)

       except Exception as e:
           print("[-] Skipping " + filename + ": " + str(e))

   if event_medians_A and event_medians_B:
       arr_A = np.array(event_medians_A)
       arr_B = np.array(event_medians_B)

       global_cv_A = (np.std(arr_A) / np.mean(arr_A)) * 100
       global_cv_B = (np.std(arr_B) / np.mean(arr_B)) * 100
       delta = abs(global_cv_A - global_cv_B)

       print("")
       print("=== Waveform Delta Global Results (per-event medians) ===")
       print("Events with IMRPhenomXPHM: " + str(len(arr_A)))
       print("Events with SEOBNRv4PHM:   " + str(len(arr_B)))
       print("IMRPhenomXPHM Global CV:   " + str(round(global_cv_A, 3)) + "%  (Baseline target: ~1.51%)")
       print("SEOBNRv4PHM Global CV:     " + str(round(global_cv_B, 3)) + "%")
       print("Approximant Delta:         " + str(round(delta, 3)) + "%")
       print("Mean n IMRPhenom:          " + str(round(float(np.mean(arr_A)), 4)) + "  (paper: 5.335)")
       print("Mean n SEOBNR:             " + str(round(float(np.mean(arr_B)), 4)))

       if delta < 0.5 and global_cv_A < 3.2 and global_cv_B < 3.2:
           print("")
           print("[+] SUCCESS: The Seraphim geometry survives GR model translation. The physics are objective.")
       elif delta < 1.0:
           print("")
           print("[+] PARTIAL: Small delta but CVs elevated. Check for NSBH events in file set.")
       else:
           print("")
           print("[!] WARNING: Significant CV deviation detected between approximants.")

   json_path = "seraphim_waveform_delta_results.json"
   csv_path = "seraphim_waveform_delta_results.csv"

   with open(json_path, "w") as jf:
       json.dump(results, jf, indent=4)

   if results:
       with open(csv_path, "w", newline="") as cf:
           writer = csv.DictWriter(cf, fieldnames=results[0].keys())
           writer.writeheader()
           writer.writerows(results)

   print("")
   print("[*] Extracted data saved to " + json_path + " and " + csv_path)


if __name__ == "__main__":
   run_waveform_delta_test()
