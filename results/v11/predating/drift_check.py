import numpy as np, time
from scipy.integrate import solve_ivp
from src.asm1.vault_loader import vault
from src.data.sensors import ObservationDataset
from src.observers import sensitivity as sens
from src.observers.reduced_model import ReducedPlantModel
from src.train.run import RAS_CHANNEL, TARGET_CHANNELS
model = ReducedPlantModel(); v = vault(); comps = v.components
ds = ObservationDataset.load("results/raw/obs_dry_sigma0p10.npz")
N = 97
t = ds.t[:N]; u = sens.InputTrajectory(ds.q_in[:N], ds.z_in[:N], ds.obs_clean[:N, ds.channels.index(RAS_CHANNEL)])
other = {"bH": 0.5 * v.p("bH"), "muH": 0.7 * v.p("muH")}
def f(tt, x):
    q, z, r = u.at(t, tt); return sens._np(model.rhs_log(x, q, z, r, params=other)).reshape(-1)
def jac(tt, x):
    q, z, r = u.at(t, tt); return sens._np(model.jacobians(x, q, z, r, params=other)[0])
x = solve_ivp(f, (t[0], t[-1]), np.log(ds.truth_reactor[0].reshape(-1)), t_eval=t, method="BDF", rtol=1e-10, atol=1e-12, jac=jac).y.T
z = np.exp(x).reshape(N, 5, 14)
tss = [c for c in TARGET_CHANNELS if c.kind == "tss_reactor"]
h = sens.log_measurement_rows(z, tss, comps)
u_sum, u_split = sens.sum_split_directions(z[0], comps)
for drift in ("model", "trajectory"):
    t0 = time.perf_counter()
    tl = sens.tangent_linear(model, x, u, t, drift=drift)
    m = sens.cumulative_propagators(tl.phis)
    F = sens.channel_fisher(None, h[:, 0], 0.01, cumulative=m)
    print(drift, "split/sum info ratio %.3e" % ((u_split @ F @ u_split) / (u_sum @ F @ u_sum)), "%.1f s" % (time.perf_counter() - t0))
