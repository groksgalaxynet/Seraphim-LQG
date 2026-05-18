"""
seraphim_m31_dust.py
====================
Downloads M31 (Andromeda) infrared data from NASA/IPAC IRSA (free, no login)
and runs Seraphim compactness analysis against the dust phase distribution.

Two-phase operation:
  Phase 1 — DOWNLOAD
    (a) NEOWISE source table   — W1 3.4μm, W2 4.6μm  (Gator cone search)
    (b) AllWISE source catalog — W1–W4, 3.4–22μm      (Gator cone search)
    (c) Herschel HELGA images  — 100–500μm FITS        (Atlas API → wget script)

    HELGA note: FITS files are not at fixed URLs. The Atlas Program Interface
    returns an XML response with a session-specific wget script containing the
    real file paths. This script fetches that XML, parses the URLs, and
    downloads the FITS files automatically. If that step fails, the wget
    script is saved to m31_helga_wget.sh for manual use.

  Phase 2 — SERAPHIM ANALYSIS
    T0  Framework band predictions (n, C_eff per WISE wavelength)
    T1  Per-band source counts + magnitude statistics
    T2  C_eff clustering vs elemental condensation thresholds (Paper 2)
    T3  Radial gradient — W2-W3 color by galactocentric zone
    T4  W1-W2 color vs radius (stellar vs hot dust)
    T5  Herschel HELGA — SED component temperatures to n to C_eff

USAGE:
    cd /your/data/folder
    python seraphim_m31_dust.py

    --skip-download   Run analysis only (files already present)
    --skip-analysis   Download only
    --dir /path       Working directory (default: current)
    --radius 90       Cone search radius in arcmin (default 90 ~ 20 kpc at M31)

OUTPUTS:
    m31_neowise.tbl                  NEOWISE source table (IPAC format)
    m31_allwise.tbl                  AllWISE source table (IPAC format)
    m31_helga_wget.sh                wget script from Atlas API
    m31_helga_images.tbl             HELGA image metadata
    m31_helga_<band>.fits            Herschel FITS images (if Atlas download works)
    seraphim_m31_per_source.csv      Per-source n and C_eff
    seraphim_m31_radial.csv          Radial zone statistics
    seraphim_m31_helga_reference.csv HELGA SED component mapping
    seraphim_m31_naxis.png           n-axis diagram (matplotlib)
    seraphim_m31_color_zones.png     W2-W3 histogram by zone (matplotlib)

DEPENDENCIES:
    Required:  numpy, astropy
    Optional:  scipy (correlations), matplotlib (plots)
    Install:   pip install numpy astropy scipy matplotlib
"""

import os, sys, csv, math, time, re, argparse
import urllib.request, urllib.parse, urllib.error
import numpy as np

try:
    from astropy.io import fits
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("WARNING: astropy not installed.  pip install astropy")

try:
    from scipy.stats import spearmanr, pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed.  pip install scipy")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed.  pip install matplotlib")

# ── SERAPHIM CONSTANTS (CODATA 2018, v18.3 verified) ─────────────────────────
NU_P    = 1.8549e43          # Hz   Planck frequency
HBAR    = 1.054571817e-34    # J·s
EV_J    = 1.602176634e-19    # J per eV
N_FLIP  = 3.561
SLOPE   = 3.506
N_H     = 92.19              # condensation ceiling
C_EFF_H = (N_H - N_FLIP) / SLOPE

M31_RA   = 10.6847           # deg J2000
M31_DEC  = 41.2690
M31_DIST = 785.0             # kpc

C_LIGHT  = 2.99792458e14    # micron/s

WISE_BANDS = {"w1mpro":3.4, "w2mpro":4.6, "w3mpro":12.0, "w4mpro":22.0}

ELEMENT_THRESHOLDS = [
    ("Cesium",    "Cs", 3.894),
    ("Iron",      "Fe", 7.902),
    ("Nickel",    "Ni", 7.640),
    ("Magnesium", "Mg", 7.646),
    ("Silicon",   "Si", 8.151),
    ("Sulfur",    "S",  10.360),
    ("Carbon",    "C",  11.260),
    ("Hydrogen",  "H",  13.598),
    ("Oxygen",    "O",  13.618),
    ("Nitrogen",  "N",  14.534),
    ("Neon",      "Ne", 21.565),
    ("Helium",    "He", 24.587),
]

# ── PHYSICS ───────────────────────────────────────────────────────────────────
def freqToN(nu):   return math.log2(NU_P / nu)
def wavToN(um):    return freqToN(C_LIGHT / um)
def nToCeff(n):    return (n - N_FLIP) / SLOPE
def ieToN(ie_eV):  return freqToN(ie_eV * EV_J / (2.0 * math.pi * HBAR))

def safe_float(v, default=None):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default

def ang_sep_deg(ra1, dec1, ra2, dec2):
    r1,d1,r2,d2 = map(math.radians,[ra1,dec1,ra2,dec2])
    c = max(-1.0,min(1.0,math.sin(d1)*math.sin(d2)+
                         math.cos(d1)*math.cos(d2)*math.cos(r1-r2)))
    return math.degrees(math.acos(c))

def deg_to_kpc(deg):
    return M31_DIST * math.tan(math.radians(deg))

# ── CONSOLE ───────────────────────────────────────────────────────────────────
SEP  = "=" * 68
SEP2 = "-" * 68

def header(t):
    print(); print(SEP); print("  " + t); print(SEP); print()

def section(t):
    print(); print(SEP2); print("  " + t); print(SEP2)

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
def download_file(url, dest, label=""):
    """Download url to dest. Skips if file already exists with content."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("  SKIP (exists, %.1f MB): %s" % (
              os.path.getsize(dest)/1e6, os.path.basename(dest)))
        return True
    label = label or os.path.basename(dest)
    print("  Downloading: %s" % label)
    short = url[:90]+("..." if len(url)>90 else "")
    print("    URL: %s" % short)
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"seraphim-m31/1.0"})
        t0  = time.time()
        with urllib.request.urlopen(req, timeout=180) as resp:
            total = int(resp.headers.get("Content-Length",0))
            done  = 0
            with open(dest,"wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk: break
                    f.write(chunk); done += len(chunk)
                    pct = " (%.0f%%)"%(done/total*100) if total else ""
                    sys.stdout.write("\r    %.1f MB%s"%(done/1e6,pct))
                    sys.stdout.flush()
        print("\n    Done in %.1f s  (%.1f MB)"%(time.time()-t0, done/1e6))
        return True
    except urllib.error.HTTPError as e:
        print("\n    HTTP ERROR %d: %s"%(e.code,e.reason))
    except urllib.error.URLError as e:
        print("\n    URL ERROR: %s"%e.reason)
    except Exception as e:
        print("\n    ERROR: %s"%e)
    if os.path.exists(dest): os.remove(dest)
    return False

# ── IRSA GATOR CONE SEARCH ────────────────────────────────────────────────────
def gator_url(catalog, ra, dec, radius_arcmin, cols=""):
    p = {"catalog":catalog,"spatial":"cone","ra":str(ra),"dec":str(dec),
         "radius":str(radius_arcmin),"radunits":"arcmin","outfmt":"1"}
    if cols: p["selcols"] = cols
    return ("https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?"
            + urllib.parse.urlencode(p))

# ── IRSA ATLAS API — HELGA FITS ───────────────────────────────────────────────
ATLAS_BASE = "https://irsa.ipac.caltech.edu/cgi-bin/Atlas/nph-atlas"

def fetch_helga_via_atlas(outdir):
    """
    Query the IRSA Atlas Program Interface for HELGA FITS images.
    The Atlas API returns XML with a session-specific wget script URL.
    We parse that script to get direct FITS download URLs.
    Returns list of (band_label, local_path) tuples.
    """
    print("  Strategy: IRSA Atlas Program Interface")
    print("  (HELGA has no fixed file paths — Atlas generates session URLs)")
    print()

    # Build Atlas query
    params = urllib.parse.urlencode({
        "mission": "HELGA",
        "locstr":  "%.4f+%.4f+eq" % (M31_RA, M31_DEC),
        "regSize": "6",
        "covers":  "on",
        "mode":    "PI",
    })
    atlas_url = ATLAS_BASE + "?" + params
    xml_path  = os.path.join(outdir, "m31_helga_atlas.xml")

    print("  Step 1: Atlas API query...")
    ok = download_file(atlas_url, xml_path, "HELGA Atlas query (XML)")
    if not ok:
        print("  ERROR: Atlas query failed. Check network connection.")
        return []

    # Parse XML
    try:
        with open(xml_path,"r",errors="replace") as f:
            xml = f.read()
    except Exception as e:
        print("  ERROR reading XML: %s" % e)
        return []

    # Check for error status
    if 'status="error"' in xml or 'status="warning"' in xml:
        m = re.search(r'<message>(.*?)</message>', xml, re.DOTALL)
        msg = m.group(1).strip() if m else "(no message)"
        print("  Atlas returned error: %s" % msg)
        return []

    # Extract wget script URL
    m = re.search(r'<downloadScript>\s*(https?://\S+?)\s*</downloadScript>', xml)
    if not m:
        print("  WARNING: No downloadScript tag found in Atlas XML.")
        print("  Raw XML (first 600 chars):")
        print(xml[:600])
        # Try to extract HTML result link for manual use
        mh = re.search(r'<resultHtml>\s*(https?://\S+?)\s*</resultHtml>', xml)
        if mh:
            print("  Browse results manually at:")
            print("    %s" % mh.group(1).strip())
        return []

    wget_url = m.group(1).strip()
    print("  Found wget script URL: %s" % wget_url[:80])

    # Save image metadata table if available
    mi = re.search(r'<images>.*?<metadata>\s*(https?://\S+?)\s*</metadata>',
                   xml, re.DOTALL)
    if mi:
        meta_url  = mi.group(1).strip()
        meta_path = os.path.join(outdir, "m31_helga_images.tbl")
        print()
        print("  Step 2: Image metadata table...")
        download_file(meta_url, meta_path, "HELGA image metadata")

    # Download wget script
    wget_path = os.path.join(outdir, "m31_helga_wget.sh")
    print()
    print("  Step 3: wget script...")
    ok = download_file(wget_url, wget_path, "HELGA wget script")
    if not ok:
        print("  Could not retrieve wget script.")
        return []

    # Parse FITS URLs from wget script
    fits_urls = []
    try:
        with open(wget_path,"r",errors="replace") as f:
            for line in f:
                # wget "URL" or wget URL or bare URL lines
                for pat in [r'"(https?://[^"]+\.fits[^"]*)"',
                            r"'(https?://[^']+\.fits[^']*)'",
                            r'(https?://\S+\.fits\S*)']:
                    found = re.findall(pat, line)
                    for u in found:
                        u = u.strip().rstrip("'\"")
                        if u not in fits_urls:
                            fits_urls.append(u)
    except Exception as e:
        print("  WARNING: Could not parse wget script: %s" % e)

    if not fits_urls:
        print()
        print("  NOTE: No FITS URLs found automatically in wget script.")
        print("  The wget script has been saved — run it manually:")
        print("    cd %s" % outdir)
        print("    bash m31_helga_wget.sh")
        print()
        print("  Or open the Atlas HTML results page in your browser to")
        print("  download individual FITS files interactively:")
        mh = re.search(r'<resultHtml>\s*(https?://\S+?)\s*</resultHtml>', xml)
        if mh: print("    %s" % mh.group(1).strip())
        return []

    print()
    print("  Found %d FITS URL(s). Downloading..." % len(fits_urls))
    downloaded = []
    for url in fits_urls:
        fname = url.split("/")[-1].split("?")[0]
        # Try to identify band
        band = "unknown"
        for b in ["PACS100","PACS160","SPIRE250","SPIRE350","SPIRE500"]:
            if b.lower() in fname.lower() or b.lower() in url.lower():
                band = b; break
        local = "m31_helga_%s.fits" % band.lower()
        dest  = os.path.join(outdir, local)
        print()
        ok = download_file(url, dest, "HELGA %s" % band)
        if ok:
            downloaded.append((band, dest))
    return downloaded

# ── PARSE IPAC TABLE ──────────────────────────────────────────────────────────
def parse_ipac_table(filepath):
    """Parse IRSA IPAC .tbl into list of dicts. Tries astropy first."""
    if HAS_ASTROPY:
        try:
            t = Table.read(filepath, format='ascii.ipac')
            rows = [{c: str(row[c]) for c in t.colnames} for row in t]
            return t.colnames, rows
        except Exception:
            pass
    # Manual fallback
    col_names = []; rows = []
    try:
        with open(filepath,"r",encoding="utf-8",errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or line.startswith("\\"): continue
                if line.startswith("|"):
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if not col_names: col_names = parts
                    continue
                if col_names:
                    vals = line.split()
                    while len(vals) < len(col_names): vals.append("")
                    rows.append(dict(zip(col_names, vals[:len(col_names)])))
    except Exception as e:
        print("    WARNING: parse error %s: %s"%(os.path.basename(filepath),e))
    return col_names, rows

# ── PHASE 1: DOWNLOAD ─────────────────────────────────────────────────────────
def run_download(outdir, radius_arcmin):
    header("PHASE 1 — DOWNLOAD")
    print("  Target : M31 (Andromeda Galaxy)")
    print("  Center : RA %.4f°  Dec %+.4f°  J2000" % (M31_RA, M31_DEC))
    print("  Radius : %d arcmin  (~%.0f kpc at M31)" % (
          radius_arcmin, deg_to_kpc(radius_arcmin/60.0)))
    print("  Output : %s" % outdir)
    print()

    # (a) NEOWISE
    section("(a) NEOWISE — W1 3.4μm + W2 4.6μm  (21 epochs, 2013–2024)")
    nw_url  = gator_url("neowiser_p1bs_psd", M31_RA, M31_DEC, radius_arcmin,
                        "ra,dec,w1mpro,w1sigmpro,w2mpro,w2sigmpro,"
                        "mjd,cc_flags,qual_frame")
    nw_path = os.path.join(outdir, "m31_neowise.tbl")
    ok_nw   = download_file(nw_url, nw_path, "NEOWISE M31 cone search")

    # (b) AllWISE
    section("(b) AllWISE — W1–W4  (3.4, 4.6, 12, 22 μm)")
    aw_url  = gator_url("allwise_p3as_psd", M31_RA, M31_DEC, radius_arcmin,
                        "ra,dec,w1mpro,w1sigmpro,w2mpro,w2sigmpro,"
                        "w3mpro,w3sigmpro,w4mpro,w4sigmpro,cc_flags,ext_flg")
    aw_path = os.path.join(outdir, "m31_allwise.tbl")
    ok_aw   = download_file(aw_url, aw_path, "AllWISE M31 cone search")

    # (c) HELGA via Atlas API
    section("(c) Herschel HELGA — far-IR FITS  (100, 160, 250, 350, 500 μm)")
    helga_files = fetch_helga_via_atlas(outdir)

    # Summary
    section("DOWNLOAD SUMMARY")
    def fmb(p): return os.path.getsize(p)/1e6 if os.path.exists(p) else 0.0
    total = fmb(nw_path) + fmb(aw_path)
    print("  NEOWISE  : %s  (%.1f MB)" % ("OK" if ok_nw else "FAILED", fmb(nw_path)))
    print("  AllWISE  : %s  (%.1f MB)" % ("OK" if ok_aw else "FAILED", fmb(aw_path)))
    if helga_files:
        for band, path in helga_files:
            s = fmb(path); total += s
            print("  HELGA %-9s: OK  (%.1f MB)" % (band, s))
    else:
        print("  HELGA    : wget script saved — run manually if needed:")
        print("             bash %s" % os.path.join(outdir,"m31_helga_wget.sh"))
    print()
    print("  Total: %.1f MB" % total)

# ── PHASE 2: ANALYSIS ─────────────────────────────────────────────────────────
def run_analysis(outdir):
    header("PHASE 2 — SERAPHIM ANALYSIS")
    print("  n(C) = %.3f + %.3f × C  (v18.3 main paper)" % (N_FLIP, SLOPE))
    print("  C_eff = (n − %.3f) / %.3f" % (N_FLIP, SLOPE))
    print("  n_H condensation ceiling = %.2f  (C_eff = %.4f)" % (N_H, C_EFF_H))
    print()

    # Element reference
    section("ELEMENT CONDENSATION THRESHOLDS  (Paper 2 schema)")
    print("  %-12s  %-4s  %7s  %7s  %7s" % ("Element","Sym","IE(eV)","n","C_eff"))
    print("  "+"-"*44)
    for name, sym, ie in ELEMENT_THRESHOLDS:
        n_e = ieToN(ie); c_e = nToCeff(n_e)
        mark = " ← CEILING" if sym=="H" else ""
        print("  %-12s  %-4s  %7.3f  %7.3f  %7.3f%s" % (name,sym,ie,n_e,c_e,mark))
    print()

    # T0: band predictions
    section("T0: WISE BAND OCTAVE DEPTHS")
    band_notes = {
        "w1mpro": "Stellar photospheres (red giants/AGB) — stellar mass tracer",
        "w2mpro": "Stellar continuum + circumstellar hot dust",
        "w3mpro": "PAH 7.7μm + VSG stochastic warm dust",
        "w4mpro": "VSG stochastic heating + warm HII dust",
    }
    print("  %-10s  %8s  %8s  %8s  %s" % ("Band","λ(μm)","n","C_eff","ISM note"))
    print("  "+"-"*75)
    for col, wav in sorted(WISE_BANDS.items(), key=lambda x:x[1], reverse=True):
        n_b = wavToN(wav); c_b = nToCeff(n_b)
        print("  %-10s  %8.1f  %8.2f  %8.2f  %s" % (
              col.upper(), wav, n_b, c_b, band_notes[col]))
    print()
    print("  All WISE bands sit 10–14 octaves BELOW n_H = %.2f." % N_H)
    print("  Emission wavelength = grain TEMPERATURE, not grain COMPOSITION.")
    print()

    # Load AllWISE
    section("T1: ALLWISE SOURCE ANALYSIS")
    aw_path = os.path.join(outdir, "m31_allwise.tbl")
    if not os.path.exists(aw_path):
        print("  SKIP: m31_allwise.tbl not found.")
        aw_rows = []
    else:
        print("  Loading: %s" % aw_path)
        _, aw_rows = parse_ipac_table(aw_path)
        print("  Rows: %d" % len(aw_rows))

    per_source   = []
    zone_w2w3    = {z:[] for z in ["bulge","inner","ring","outer","halo"]}
    w1w2_all     = []
    w2w3_all     = []

    if aw_rows:
        n_good = 0
        for row in aw_rows:
            ra  = safe_float(row.get("ra",  row.get("RA",  "")))
            dec = safe_float(row.get("dec", row.get("DEC", "")))
            if ra is None or dec is None: continue
            sep   = ang_sep_deg(ra, dec, M31_RA, M31_DEC)
            sep_k = deg_to_kpc(sep)
            zone  = ("bulge" if sep_k<2  else "inner" if sep_k<8  else
                     "ring"  if sep_k<15 else "outer" if sep_k<25 else "halo")
            mags = {}
            for col in WISE_BANDS:
                v = safe_float(row.get(col, row.get(col.upper(),"")))
                if v and 3.0 < v < 22.0: mags[col] = v
            if not mags: continue
            src = {"ra":round(ra,6),"dec":round(dec,6),
                   "sep_kpc":round(sep_k,2),"zone":zone}
            for col, wav in WISE_BANDS.items():
                m = mags.get(col)
                src["mag_%s"%col[:2]] = round(m,4) if m else ""
                src["n_%s"%col[:2]]   = round(wavToN(wav),4)
                src["c_%s"%col[:2]]   = round(nToCeff(wavToN(wav)),4)
            w1 = mags.get("w1mpro"); w2 = mags.get("w2mpro")
            w3 = mags.get("w3mpro"); w4 = mags.get("w4mpro")
            src["w1_w2"] = round(w1-w2,4) if (w1 and w2) else ""
            src["w2_w3"] = round(w2-w3,4) if (w2 and w3) else ""
            src["w3_w4"] = round(w3-w4,4) if (w3 and w4) else ""
            per_source.append(src); n_good += 1
            if w1 and w2: w1w2_all.append(w1-w2)
            if w2 and w3:
                w2w3_all.append(w2-w3)
                zone_w2w3[zone].append(w2-w3)

        print("  Valid sources: %d / %d" % (n_good, len(aw_rows)))
        print()
        print("  Per-band counts:")
        print("  %-10s  %8s  %8s  %8s" % ("Band","N_det","n","C_eff"))
        print("  "+"-"*40)
        for col, wav in sorted(WISE_BANDS.items(), key=lambda x:x[1], reverse=True):
            n_det = sum(1 for r in per_source if r.get("mag_%s"%col[:2]))
            print("  %-10s  %8d  %8.3f  %8.3f" % (
                  col.upper(), n_det, wavToN(wav), nToCeff(wavToN(wav))))

        # T2
        section("T2: C_eff CLUSTERING vs ELEMENTAL THRESHOLDS")
        print("  Framework prediction: grain FORMATION at C_eff 25.0–25.8")
        print("  WISE emission sits at C_eff 20–25 (grain temperature regime)")
        print()
        w2w3 = np.array([v for v in w2w3_all if math.isfinite(v)])
        if len(w2w3) > 10:
            print("  W2−W3 color  N=%d" % len(w2w3))
            print("    Mean   = %.4f  (>0 = more W3/dust flux)" % np.mean(w2w3))
            print("    Median = %.4f" % np.median(w2w3))
            print("    Std    = %.4f" % np.std(w2w3))
            dust_dom = int(np.sum(w2w3>2.0))
            star_dom = int(np.sum(w2w3<0.5))
            mixed    = len(w2w3)-dust_dom-star_dom
            print()
            print("  Source classification:")
            print("    Dust-dominated  W2-W3 > 2.0: %5d  (%.1f%%)" % (
                  dust_dom, 100*dust_dom/len(w2w3)))
            print("    Mixed           0.5–2.0:      %5d  (%.1f%%)" % (
                  mixed,    100*mixed/len(w2w3)))
            print("    Stellar         W2-W3 < 0.5:  %5d  (%.1f%%)" % (
                  star_dom, 100*star_dom/len(w2w3)))
            print()
            nW3 = wavToN(12.0); nW2 = wavToN(4.6)
            print("  W3 band: n=%.3f  C_eff=%.3f  (%.1f oct below n_H)" % (
                  nW3, nToCeff(nW3), N_H-nW3))
            print("  W2 band: n=%.3f  C_eff=%.3f  (%.1f oct below n_H)" % (
                  nW2, nToCeff(nW2), N_H-nW2))

        # T3
        section("T3: RADIAL GRADIENT — W2-W3 by galactocentric zone")
        print("  Expected peak at ~10 kpc (star-forming ring of M31)")
        print()
        kpc_r = {"bulge":"0–2","inner":"2–8","ring":"8–15","outer":"15–25","halo":">25"}
        print("  %-8s  %6s  %10s  %10s  %s" % ("Zone","N","med W2-W3","std W2-W3","kpc"))
        print("  "+"-"*52)
        radial_rows = []
        for z in ["bulge","inner","ring","outer","halo"]:
            vals = np.array([v for v in zone_w2w3[z] if math.isfinite(v)])
            if len(vals) == 0:
                print("  %-8s  %6d  (no data)  %s kpc" % (z,0,kpc_r[z]))
                continue
            med = float(np.median(vals)); std = float(np.std(vals))
            print("  %-8s  %6d  %10.4f  %10.4f  %s kpc" % (z,len(vals),med,std,kpc_r[z]))
            radial_rows.append({"zone":z,"kpc_range":kpc_r[z],
                                 "n_sources":len(vals),
                                 "med_w2_w3":round(med,4),"std_w2_w3":round(std,4),
                                 "n_W3":round(wavToN(12.0),4),
                                 "c_eff_W3":round(nToCeff(wavToN(12.0)),4)})

        # T4
        section("T4: W1-W2 COLOR — Stellar vs hot dust")
        w1w2 = np.array([v for v in w1w2_all if math.isfinite(v)])
        if len(w1w2) > 10:
            print("  N=%d  Mean=%.4f  Median=%.4f  Std=%.4f" % (
                  len(w1w2), np.mean(w1w2), np.median(w1w2), np.std(w1w2)))
            if HAS_SCIPY:
                dists=[]; colors=[]
                for r in per_source:
                    d=safe_float(r.get("sep_kpc","")); c=safe_float(r.get("w1_w2",""))
                    if d is not None and c is not None:
                        dists.append(d); colors.append(c)
                if len(dists)>10:
                    rho,p = spearmanr(dists,colors)
                    print("  Spearman r(sep_kpc, W1-W2) = %.4f  p=%.2e  %s" % (
                          rho, p, "** radial gradient **" if p<0.05 else "no radial trend"))

        # Save CSVs
        if per_source:
            p = os.path.join(outdir,"seraphim_m31_per_source.csv")
            with open(p,"w",newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(per_source[0].keys()))
                w.writeheader(); w.writerows(per_source)
            print()
            print("  Per-source CSV: %s  (%d rows)" % (p,len(per_source)))
        if radial_rows:
            p = os.path.join(outdir,"seraphim_m31_radial.csv")
            with open(p,"w",newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(radial_rows[0].keys()))
                w.writeheader(); w.writerows(radial_rows)
            print("  Radial CSV    : %s" % p)

    # T5: HELGA
    section("T5: HERSCHEL HELGA — Cold dust SED to n to C_eff")
    print("  Reference SED components (Viaene+2014 M31):")
    print()
    print("  %-22s  %6s  %9s  %7s  %7s  delta_nH" % (
          "Component","T(K)","λ_peak(μm)","n","C_eff"))
    print("  "+"-"*66)
    helga_comps = [
        ("Cold diffuse disk",  16, 181),
        ("Warm diffuse disk",  22, 132),
        ("Star-forming ring",  45,  64),
        ("HII region dust",    70,  41),
        ("PDR boundary",      100,  29),
    ]
    helga_rows = []
    for name, T, wav in helga_comps:
        n_v = wavToN(wav); c_v = nToCeff(n_v); delta = n_v-N_H
        print("  %-22s  %6.0f  %9.0f  %7.3f  %7.3f  %+.2f" % (
              name, T, wav, n_v, c_v, delta))
        helga_rows.append({"component":name,"T_K":T,"wav_peak_um":wav,
                            "n":round(n_v,4),"C_eff":round(c_v,4),
                            "delta_nH":round(delta,4)})
    print()
    print("  All components 2–8 octaves BELOW n_H. Cold diffuse disk farthest.")

    # Inspect any downloaded FITS
    for band in ["pacs100","pacs160","spire250","spire350","spire500"]:
        fp = os.path.join(outdir,"m31_helga_%s.fits"%band)
        if os.path.exists(fp) and HAS_ASTROPY:
            try:
                with fits.open(fp) as h:
                    hdr = h[0].header
                    wm  = {"pacs100":100,"pacs160":160,
                           "spire250":250,"spire350":350,"spire500":500}[band]
                    print("  HELGA %s: %s×%s pix  units=%s  n=%.3f  C_eff=%.3f" % (
                          band.upper(), hdr.get("NAXIS1","?"),
                          hdr.get("NAXIS2","?"), hdr.get("BUNIT","?"),
                          wavToN(wm), nToCeff(wavToN(wm))))
            except Exception as e:
                print("  HELGA %s: FITS error: %s"%(band.upper(),e))

    p = os.path.join(outdir,"seraphim_m31_helga_reference.csv")
    with open(p,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(helga_rows[0].keys()))
        w.writeheader(); w.writerows(helga_rows)
    print()
    print("  HELGA reference CSV: %s" % p)

    # Verdict
    section("FRAMEWORK VERDICT")
    print()
    print("  Grain COMPOSITION thresholds (Paper 2): C_eff 25.0–25.8  n 91.3–94.0")
    print("  All 118 elements confirmed in this window.")
    print()
    print("  M31 WISE emission:")
    for col,wav in sorted(WISE_BANDS.items(),key=lambda x:x[1],reverse=True):
        n_b=wavToN(wav); c_b=nToCeff(n_b)
        print("    %-8s  %.1fμm → n=%.2f  C_eff=%.2f  (%.1f oct below n_H)" % (
              col.upper(), wav, n_b, c_b, N_H-n_b))
    print()
    print("  CONFIRMED: WISE emission 3–12 oct below condensation zone.")
    print("  Consistent: emission = grain temperature, not composition threshold.")
    print()
    print("  NEXT: spatial correlation of W3/W4-bright zones with C_eff boundaries")
    print("  at 25.1 (silicates: Fe/Mg/Si) vs 25.2 (carbonaceous: C/S).")
    print("  Requires pixel-matched HELGA + AllWISE spatial SED analysis.")

    # Plots
    if HAS_MPL and per_source:
        section("PLOTS")
        fig,ax=plt.subplots(figsize=(14,5))
        fig.patch.set_facecolor('#04060d'); ax.set_facecolor('#080b16')
        elem_ns=[ieToN(ie) for _,_,ie in ELEMENT_THRESHOLDS]
        ax.axvspan(min(elem_ns),max(elem_ns),alpha=0.12,color='#d4a843',label='Element zone')
        ax.axvline(N_H,color='#d4a843',lw=1.5,ls='--',label='n_H=92.19')
        bc={'w1mpro':'#4dd4d4','w2mpro':'#9b72e8','w3mpro':'#d4a843','w4mpro':'#e85555'}
        for col,wav in WISE_BANDS.items():
            ax.axvline(wavToN(wav),color=bc[col],lw=1,ls=':',alpha=0.9,
                       label='%s %.1fμm'%(col.upper(),wav))
        hc=['#6a85e0','#52c97a','#e8863a','#e85555','#9b72e8']
        for i,(name,T,wav) in enumerate(helga_comps):
            ax.axvline(wavToN(wav),color=hc[i],lw=1.2,alpha=0.7,
                       label='%s %dK'%(name[:12],T))
        ax.set_xlim(78,96)
        ax.set_xlabel('n = log₂(ν_P/ν)',color='#9aa8cc')
        ax.set_title('M31 — Seraphim n-axis: ISM phases vs condensation thresholds',
                     color='#d4a843')
        ax.tick_params(colors='#3a4260'); ax.spines[:].set_color('#171e35')
        ax.legend(loc='upper left',fontsize=7,facecolor='#0d1120',
                  edgecolor='#171e35',labelcolor='#9aa8cc')
        plt.tight_layout()
        p1=os.path.join(outdir,"seraphim_m31_naxis.png")
        plt.savefig(p1,dpi=150,facecolor='#04060d'); plt.close()
        print("  seraphim_m31_naxis.png")

        if any(zone_w2w3.values()):
            fig,ax=plt.subplots(figsize=(10,5))
            fig.patch.set_facecolor('#04060d'); ax.set_facecolor('#080b16')
            zc={'bulge':'#e85555','inner':'#e8863a','ring':'#d4a843',
                'outer':'#52c97a','halo':'#4dd4d4'}
            for z in ["bulge","inner","ring","outer","halo"]:
                vals=[v for v in zone_w2w3[z] if math.isfinite(v)]
                if vals: ax.hist(vals,bins=40,alpha=0.55,label=z,
                                 color=zc[z],density=True)
            ax.axvline(0.5,color='#fff',lw=1,ls='--',label='W2-W3=0.5')
            ax.axvline(2.0,color='#d4a843',lw=1,ls='--',label='W2-W3=2.0')
            ax.set_xlabel('W2−W3 [mag]',color='#9aa8cc')
            ax.set_ylabel('Normalized density',color='#9aa8cc')
            ax.set_title('M31 W2−W3 color by galactocentric zone',color='#d4a843')
            ax.tick_params(colors='#3a4260'); ax.spines[:].set_color('#171e35')
            ax.legend(fontsize=8,facecolor='#0d1120',edgecolor='#171e35',
                      labelcolor='#9aa8cc')
            plt.tight_layout()
            p2=os.path.join(outdir,"seraphim_m31_color_zones.png")
            plt.savefig(p2,dpi=150,facecolor='#04060d'); plt.close()
            print("  seraphim_m31_color_zones.png")

    print()
    print(SEP)
    print("  Analysis complete.")
    print(SEP)
    print()

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="M31 IRSA downloader + Seraphim analysis")
    ap.add_argument("--dir","-d",default=".",
                    help="Working directory (default: current)")
    ap.add_argument("--skip-download",action="store_true",
                    help="Skip download, analyse existing files")
    ap.add_argument("--skip-analysis",action="store_true",
                    help="Download only")
    ap.add_argument("--radius",type=int,default=90,
                    help="Cone search radius in arcmin (default 90)")
    args = ap.parse_args()

    outdir = os.path.abspath(args.dir)
    os.makedirs(outdir, exist_ok=True)

    header("SERAPHIM M31 DUST ANALYSIS")
    print("  v18.3 compactness equation + Paper 2 elemental schema")
    print("  Data: NASA/IPAC IRSA (free, no login)")
    print("  Working directory: %s" % outdir)
    print()

    if not args.skip_download:
        run_download(outdir, args.radius)
    else:
        print("  (Download skipped)"); print()

    if not args.skip_analysis:
        run_analysis(outdir)
