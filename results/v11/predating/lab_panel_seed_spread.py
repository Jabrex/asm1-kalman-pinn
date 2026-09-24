import json, numpy as np, sys
sys.path.insert(0, ".")
from src.asm1.vault_loader import vault
from src.asm1.truth_plants import truth_vault
from src.data.sensors import ObservationDataset
from src.observers import anchors as A
import scripts.make_anchors as M

ens = np.load("results/v11/anchors/nominal_ensemble.npz")
Xe = np.log(np.maximum(ens["members"][ens["accepted"]].reshape(-1, 70), 1e-12))
C = np.cov(Xe, rowvar=False)
ops_n = A.lab_operators(vault()); names = list(M.PANEL_ERRORS); errs = np.array(list(M.PANEL_ERRORS.values()))
W = np.stack([ops_n[k].reshape(-1) for k in names]); R = np.diag(np.log1p(errs**2))

def full_update(mean, y, P):
    x = np.log(np.maximum(mean.reshape(-1), 1e-12)); z = np.exp(x); yp = W @ z
    H = W * z[None, :] / yp[:, None]
    K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
    return np.exp(x + K @ (np.log(y) - np.log(yp))).reshape(5, 14)

for kin in ["k000", "k025", "k050", "k075", "k100", "k000_off"]:
    ddir = {"k000": "results/raw"}.get(kin, "results/raw_" + kin)
    dry = ObservationDataset.load(ddir + "/obs_dry_sigma0p00.npz"); zt = dry.truth_reactor[0]
    tv = truth_vault(str(dry.meta.get("truth_preset") or "vault20"), float(dry.meta["alpha"]) if dry.meta.get("alpha") is not None else 1.0)
    ops_t = A.lab_operators(tv)
    As = np.load("results/v11/anchors/%s/As.npz" % kin)
    rms = lambda m: float(np.sqrt(np.mean(np.log(np.maximum(m, 1e-12) / np.maximum(zt, 1e-12))**2)))
    res = {"diag": [], "full": [], "full_shrunk": []}
    Pd = np.diag(np.maximum(np.log1p(As["z0_rel_std"].reshape(-1)**2), 1e-12))
    Ps = 0.8 * C + 0.2 * np.diag(np.diag(C))
    for seed in range(20):
        rng = np.random.default_rng([M.LAB_SEED, 100 + seed]) if seed else np.random.default_rng(M.LAB_SEED)
        panel = A.lab_values(ops_t, zt, M.PANEL_ERRORS, rng); y = np.array([panel[k] for k in names])
        res["diag"].append(rms(full_update(As["z0_mean"], y, Pd)))
        res["full"].append(rms(full_update(As["z0_mean"], y, C)))
        res["full_shrunk"].append(rms(full_update(As["z0_mean"], y, Ps)))
    print("%-9s As %.3f | diag seed0 %.3f mean20 %.3f | full seed0 %.3f mean20 %.3f | shrunk seed0 %.3f mean20 %.3f"
          % (kin, rms(As["z0_mean"]), res["diag"][0], np.mean(res["diag"]), res["full"][0], np.mean(res["full"]),
             res["full_shrunk"][0], np.mean(res["full_shrunk"])))
