import numpy as np
from fenics import *

beta_groups_base = np.array([0.33, 0.44, 0.495, 0.44, 0.385, 0.275, 0.198])
sigma_groups = np.array([0.25, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10])
gamma_groups_base = np.array([0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04])
alpha_groups = np.array([0.00002, 0.0001, 0.0003, 0.003, 0.01, 0.035, 0.09])
population_distribution = np.array([0.225, 0.040, 0.175, 0.245, 0.175, 0.095, 0.045])
epsilon_groups = np.array([0.92, 0.92, 0.90, 0.88, 0.85, 0.80, 0.75])
D_base_groups = np.array([8.0, 10.0, 12.0, 7.0, 5.0, 3.0, 1.5])
exposed_distribution = np.array([0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05])
contact_matrix = np.array([
    [1.2,  0.8,  0.3,  0.2,  0.1,  0.05, 0.02],
    [0.8,  1.5,  0.9,  0.4,  0.2,  0.1,  0.05],
    [0.3,  0.9,  1.3,  0.8,  0.3,  0.2,  0.1 ],
    [0.2,  0.4,  0.8,  1.2,  0.6,  0.3,  0.2 ],
    [0.1,  0.2,  0.3,  0.6,  1.1,  0.5,  0.3 ],
    [0.05, 0.1,  0.2,  0.3,  0.5,  1.0,  0.6 ],
    [0.02, 0.05, 0.1,  0.2,  0.3,  0.6,  0.9 ],
])

N = 367000.0
domain_l = 100.0
focus_frac = 0.15
focus_sigma_frac = 0.06

VACC_STRATEGY_UNIFORM = np.array([0.005] * 7)
VACC_STRATEGY_ELDERLY_PRIORITY = np.array(
    [0.003, 0.004, 0.004, 0.005, 0.008, 0.012, 0.015])
VACC_STRATEGY_YOUTH_PRIORITY = np.array(
    [0.015, 0.012, 0.008, 0.005, 0.004, 0.004, 0.003])


class GaussianFocus(UserExpression):
    def __init__(self, amplitude, x0, y0, sigma, **kwargs):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.x0, self.y0, self.sigma = x0, y0, sigma

    def eval(self, value, x):
        r2 = (x[0] - self.x0) ** 2 + (x[1] - self.y0) ** 2
        value[0] = self.amplitude * np.exp(-r2 / (2.0 * self.sigma ** 2))

    def value_shape(self):
        return ()


class UniformBackground(UserExpression):
    def __init__(self, background, focus_expr, **kwargs):
        super().__init__(**kwargs)
        self.background = background
        self.focus_expr = focus_expr

    def eval(self, value, x):
        fval = np.zeros(1)
        self.focus_expr.eval(fval, x)
        value[0] = max(self.background - fval[0], 0.0)

    def value_shape(self):
        return ()


def run_simulation(mesh_n=50, dt=0.5, T=160.0, beta_groups=None, gamma_groups=None,
                   upsilon_groups=None, dose_cap_total=None, verbose=False,
                   record_series=False, snapshot_every_days=None):

    if beta_groups is None:
        beta_groups = beta_groups_base
    if gamma_groups is None:
        gamma_groups = gamma_groups_base
    if upsilon_groups is None:
        upsilon_groups = VACC_STRATEGY_UNIFORM
    beta_groups = np.asarray(beta_groups, dtype=float)
    gamma_groups = np.asarray(gamma_groups, dtype=float)
    upsilon_groups = np.asarray(upsilon_groups, dtype=float)

    n_age = len(beta_groups)
    domain_area = domain_l ** 2

    D_S_groups = D_base_groups * 1.00
    D_V_groups = D_base_groups * 1.00
    D_E_groups = D_base_groups * 0.80
    D_I_groups = D_base_groups * 0.40
    D_R_groups = D_base_groups * 0.90

    mesh = RectangleMesh(Point(0, 0), Point(domain_l, domain_l), mesh_n, mesh_n)
    FS = FunctionSpace(mesh, 'P', 1)
    v = TestFunction(FS)

    S_groups  = [Function(FS) for _ in range(n_age)]
    Vc_groups = [Function(FS) for _ in range(n_age)]
    E_groups  = [Function(FS) for _ in range(n_age)]
    I_groups  = [Function(FS) for _ in range(n_age)]
    R_groups  = [Function(FS) for _ in range(n_age)]
    Dd_groups = [Function(FS) for _ in range(n_age)]

    S_n  = [Function(FS) for _ in range(n_age)]
    Vc_n = [Function(FS) for _ in range(n_age)]
    E_n  = [Function(FS) for _ in range(n_age)]
    I_n  = [Function(FS) for _ in range(n_age)]
    R_n  = [Function(FS) for _ in range(n_age)]
    Dd_n = [Function(FS) for _ in range(n_age)]

    FOCUS_X, FOCUS_Y = 0.5 * domain_l, 0.5 * domain_l
    FOCUS_SIGMA = focus_sigma_frac * domain_l

    for i in range(n_age):
        N_group = N * population_distribution[i]
        amp_I = focus_frac * N_group * exposed_distribution[i]
        amp_E = amp_I * 0.5
        I_expr = GaussianFocus(amp_I, FOCUS_X, FOCUS_Y, FOCUS_SIGMA, degree=2)
        E_expr = GaussianFocus(amp_E, FOCUS_X, FOCUS_Y, FOCUS_SIGMA, degree=2)
        S_expr = UniformBackground(N_group, I_expr, degree=2)

        S_groups[i].assign(interpolate(S_expr, FS))
        E_groups[i].assign(interpolate(E_expr, FS))
        I_groups[i].assign(interpolate(I_expr, FS))
        Vc_groups[i].assign(interpolate(Constant(0.0), FS))
        R_groups[i].assign(interpolate(Constant(0.0), FS))
        Dd_groups[i].assign(interpolate(Constant(0.0), FS))

        S_n[i].assign(S_groups[i]);  Vc_n[i].assign(Vc_groups[i])
        E_n[i].assign(E_groups[i]);  I_n[i].assign(I_groups[i])
        R_n[i].assign(R_groups[i]);  Dd_n[i].assign(Dd_groups[i])

    def force_of_infection(idx):
        lam_field = Function(FS)
        lam_vals = np.zeros(FS.dim())
        for j in range(n_age):
            coeff = beta_groups[idx] * contact_matrix[idx, j] / N
            lam_vals += coeff * I_groups[j].vector().get_local()
        lam_field.vector().set_local(lam_vals)
        lam_field.vector().apply('insert')
        return lam_field

    total_mass_t0 = sum(
        assemble((S_groups[i] + Vc_groups[i] + E_groups[i]
                  + I_groups[i] + R_groups[i] + Dd_groups[i]) * dx)
        for i in range(n_age)
    ) / domain_area

    upsilon_active = upsilon_groups.copy()
    Q_cumulative = 0.0
    dose_capped = False
    global_min = {k: np.inf for k in ['S', 'V', 'E', 'I', 'R', 'D']}

    time_points, I_total_series = [], []
    D_group_values = [[] for _ in range(n_age)]

    series_times, series_totals = [], []
    snap_times, snap_I = [], []
    snap_steps = (max(1, int(round(snapshot_every_days / dt)))
                  if snapshot_every_days is not None else None)

    def save_state(t_now, step_idx):
        if not record_series:
            return
        groups = (S_groups, Vc_groups, E_groups, I_groups, R_groups, Dd_groups)
        series_times.append(t_now)
        series_totals.append(np.array([[assemble(g[i] * dx) / domain_area for i in range(n_age)]
                                       for g in groups]))
        if snap_steps is not None and step_idx % snap_steps == 0:
            snap_times.append(t_now)
            snap_I.append(np.array([I_groups[i].vector().get_local() for i in range(n_age)]))

    step_idx = 0
    save_state(0.0, 0)
    t = 0.0
    while t <= T:
        for i in range(n_age):
            lam = force_of_infection(i)
            ups_i = Constant(upsilon_active[i])
            eps_i = Constant(epsilon_groups[i])
            DS_i = Constant(D_S_groups[i]); DV_i = Constant(D_V_groups[i])
            DE_i = Constant(D_E_groups[i]); DI_i = Constant(D_I_groups[i])
            DR_i = Constant(D_R_groups[i])

            F_S = ((S_groups[i] - S_n[i]) / dt * v
                   + lam * S_groups[i] * v
                   + ups_i * S_groups[i] * v
                   + DS_i * inner(grad(S_groups[i]), grad(v))) * dx

            F_V = ((Vc_groups[i] - Vc_n[i]) / dt * v
                   - ups_i * S_n[i] * v
                   + (1 - eps_i) * lam * Vc_groups[i] * v
                   + DV_i * inner(grad(Vc_groups[i]), grad(v))) * dx

            F_E = ((E_groups[i] - E_n[i]) / dt * v
                   - lam * S_n[i] * v
                   - (1 - eps_i) * lam * Vc_n[i] * v
                   + sigma_groups[i] * E_groups[i] * v
                   + DE_i * inner(grad(E_groups[i]), grad(v))) * dx

            F_I = ((I_groups[i] - I_n[i]) / dt * v
                   - sigma_groups[i] * E_n[i] * v
                   + (gamma_groups[i] + alpha_groups[i]) * I_groups[i] * v
                   + DI_i * inner(grad(I_groups[i]), grad(v))) * dx

            F_R = ((R_groups[i] - R_n[i]) / dt * v
                   - gamma_groups[i] * I_n[i] * v
                   + DR_i * inner(grad(R_groups[i]), grad(v))) * dx

            F_D = ((Dd_groups[i] - Dd_n[i]) / dt * v
                   - alpha_groups[i] * I_n[i] * v) * dx

            solve(F_S == 0, S_groups[i]); solve(F_V == 0, Vc_groups[i])
            solve(F_E == 0, E_groups[i]); solve(F_I == 0, I_groups[i])
            solve(F_R == 0, R_groups[i]); solve(F_D == 0, Dd_groups[i])

            for name, func in (('S', S_groups[i]), ('V', Vc_groups[i]), ('E', E_groups[i]),
                               ('I', I_groups[i]), ('R', R_groups[i]), ('D', Dd_groups[i])):
                m = float(func.vector().get_local().min())
                if m < global_min[name]:
                    global_min[name] = m

        dose_step = dt * sum(assemble(Constant(upsilon_active[i]) * S_n[i] * dx)
                              for i in range(n_age)) / domain_area
        Q_cumulative += dose_step
        if (dose_cap_total is not None) and (not dose_capped) and (Q_cumulative >= dose_cap_total):
            upsilon_active = np.zeros(n_age)
            dose_capped = True
            if verbose:
                print(f"  [t={t:.1f}] бюджет доз исчерпан (Q={Q_cumulative:.0f}), вакцинация остановлена")

        time_points.append(t)
        I_total_series.append(sum(assemble(I_groups[i] * dx) for i in range(n_age)) / domain_area)
        for i in range(n_age):
            D_group_values[i].append(assemble(Dd_groups[i] * dx) / domain_area)
        step_idx += 1
        save_state(step_idx * dt, step_idx)

        for i in range(n_age):
            S_n[i].assign(S_groups[i]);  Vc_n[i].assign(Vc_groups[i])
            E_n[i].assign(E_groups[i]);  I_n[i].assign(I_groups[i])
            R_n[i].assign(R_groups[i]);  Dd_n[i].assign(Dd_groups[i])

        t += dt

    total_deaths = sum(D_group_values[i][-1] for i in range(n_age))

    total_mass_tT = sum(
        assemble((S_groups[i] + Vc_groups[i] + E_groups[i]
                  + I_groups[i] + R_groups[i] + Dd_groups[i]) * dx)
        for i in range(n_age)
    ) / domain_area

    if verbose:
        print(f"  [проверка массы] N задано = {N:.0f}")
        print(f"  [проверка массы] сумма всех компартментов при t=0: {total_mass_t0:.0f} "
              f"(отношение к N: {total_mass_t0 / N:.4f})")
        print(f"  [проверка массы] сумма всех компартментов при t=T: {total_mass_tT:.0f} "
              f"(отношение к N: {total_mass_tT / N:.4f})")

    result = {
        'total_deaths': total_deaths,
        'D_by_group': np.array([D_group_values[i][-1] for i in range(n_age)]),
        'Q_cumulative': Q_cumulative,
        'total_mass_t0': total_mass_t0,
        'total_mass_tT': total_mass_tT,
        'time_points': np.array(time_points),
        'I_total_series': np.array(I_total_series),
        'min_values': global_min,
    }
    if record_series:
        result['series_times'] = np.array(series_times)
        result['series_totals'] = np.array(series_totals)
        if snap_steps is not None:
            result['snap_times'] = np.array(snap_times)
            result['snap_I'] = np.array(snap_I)
            result['dof_xy'] = FS.tabulate_dof_coordinates().reshape(-1, 2)
    return result
