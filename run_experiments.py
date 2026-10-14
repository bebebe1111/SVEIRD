import csv
import math
import os

from model_core import (run_simulation, VACC_STRATEGY_UNIFORM,
                        VACC_STRATEGY_ELDERLY_PRIORITY, VACC_STRATEGY_YOUTH_PRIORITY,
                        beta_groups_base, gamma_groups_base)


def save_row(path, fieldnames, row, first=False):
    with open(path, 'w' if first else 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if first:
            writer.writeheader()
        writer.writerow(row)


def convergence_study(output_dir='.', mesh_levels=(25, 50, 100),
                      dt_levels=(0.5, 0.25, 0.125), T=160.0):
    path = os.path.join(output_dir, 'convergence_results.csv')
    fieldnames = ['series', 'h_km', 'dt_days', 'total_deaths', 'deviation_pct',
                  'min_S', 'min_V', 'min_E', 'min_I', 'min_R', 'min_D']

    base_mesh_n, base_dt = mesh_levels[1], dt_levels[0]
    configs = ([('по h (dt фиксирован)', m, base_dt) for m in mesh_levels]
               + [('по dt (h фиксирован)', base_mesh_n, d) for d in dt_levels])

    results = {}
    base_deaths = None
    for k, (series, mesh_n, dt) in enumerate(configs):
        print(f"[{series}] h={100.0 / mesh_n:.2f} км, dt={dt} сут ...")
        if (mesh_n, dt) not in results:
            results[(mesh_n, dt)] = run_simulation(mesh_n=mesh_n, dt=dt, T=T,
                                                   upsilon_groups=VACC_STRATEGY_UNIFORM,
                                                   verbose=True)
        res = results[(mesh_n, dt)]
        mv = res['min_values']
        print(f"  смертность к T={T}: {res['total_deaths']:.1f} чел.")
        if base_deaths is None:
            base_deaths = res['total_deaths']
        row = {'series': series, 'h_km': 100.0 / mesh_n, 'dt_days': dt,
               'total_deaths': res['total_deaths'],
               'deviation_pct': abs(res['total_deaths'] - base_deaths) / base_deaths * 100}
        for name in 'SVEIRD':
            row['min_' + name] = mv[name]
        save_row(path, fieldnames, row, first=(k == 0))


def calibrate_strategy_scale(ups, Q_target, mesh_n, dt, T, rel_tol=0.003, max_iter=10):
    def q_of(k):
        return run_simulation(mesh_n=mesh_n, dt=dt, T=T, upsilon_groups=ups * k)['Q_cumulative']

    k0, q0 = 1.0, q_of(1.0)
    print(f"    k={k0:.4f}: Q={q0:.0f} (цель {Q_target:.0f})")
    if abs(q0 - Q_target) / Q_target < rel_tol:
        return k0
    k1 = k0 * (Q_target / q0)
    q1 = q_of(k1)
    print(f"    k={k1:.4f}: Q={q1:.0f}")
    for _ in range(max_iter):
        if abs(q1 - Q_target) / Q_target < rel_tol:
            return k1
        slope = (math.log(q1) - math.log(q0)) / (math.log(k1) - math.log(k0))
        k2 = math.exp(math.log(k1) + (math.log(Q_target) - math.log(q1)) / slope)
        k0, q0 = k1, q1
        k1, q1 = k2, q_of(k2)
        print(f"    k={k1:.4f}: Q={q1:.0f}")
    print("    ВНИМАНИЕ: калибровка не сошлась, берём последнее приближение")
    return k1


def calibrated_strategies(Q_target, mesh_n, dt, T):
    base = {'Равномерная': VACC_STRATEGY_UNIFORM,
            'Приоритет пожилых': VACC_STRATEGY_ELDERLY_PRIORITY,
            'Приоритет молодых': VACC_STRATEGY_YOUTH_PRIORITY}
    strategies, scales = {}, {}
    for name, ups in base.items():
        print(f"  калибровка «{name}»...")
        k = calibrate_strategy_scale(ups, Q_target, mesh_n, dt, T)
        strategies[name], scales[name] = ups * k, k
    return strategies, scales


def dose_equalized_strategies(mesh_n=50, dt=0.5, T=160.0, output_dir='.'):
    print("Базовый прогон: определяем Q_target...")
    base = run_simulation(mesh_n=mesh_n, dt=dt, T=T, upsilon_groups=VACC_STRATEGY_UNIFORM)
    Q_target = base['Q_cumulative']
    print(f"Q_target = {Q_target:.0f} доз")

    strategies, scales = calibrated_strategies(Q_target, mesh_n, dt, T)
    path = os.path.join(output_dir, 'dose_equalized_strategies.csv')
    fieldnames = ['strategy', 'rate_scale_k', 'Q_cumulative', 'total_deaths']
    for k, (name, ups) in enumerate(strategies.items()):
        print(f"Расчёт: стратегия «{name}»...")
        res = run_simulation(mesh_n=mesh_n, dt=dt, T=T, upsilon_groups=ups,
                             dose_cap_total=Q_target)
        print(f"  смертность: {res['total_deaths']:.1f} чел., доз: {res['Q_cumulative']:.0f}")
        row = {'strategy': name, 'rate_scale_k': scales[name],
               'Q_cumulative': res['Q_cumulative'], 'total_deaths': res['total_deaths']}
        save_row(path, fieldnames, row, first=(k == 0))


def sensitivity_analysis(mesh_n=50, dt=0.5, T=160.0, output_dir='.'):
    base = run_simulation(mesh_n=mesh_n, dt=dt, T=T, upsilon_groups=VACC_STRATEGY_UNIFORM)
    Q_target = base['Q_cumulative']
    strategies, _ = calibrated_strategies(Q_target, mesh_n, dt, T)

    perturbations = {'baseline':  (1.00, 1.00),
                     'beta+10%':  (1.10, 1.00),
                     'beta-10%':  (0.90, 1.00),
                     'gamma+10%': (1.00, 1.10),
                     'gamma-10%': (1.00, 0.90)}
    path = os.path.join(output_dir, 'sensitivity_results.csv')
    names = list(strategies)
    fieldnames = ['perturbation'] + names + ['Q_' + n for n in names] + ['order']

    for k, (pname, (b_scale, g_scale)) in enumerate(perturbations.items()):
        deaths, doses = {}, {}
        for sname, ups in strategies.items():
            print(f"[{pname}] стратегия «{sname}»...")
            res = run_simulation(mesh_n=mesh_n, dt=dt, T=T, upsilon_groups=ups,
                                 beta_groups=beta_groups_base * b_scale,
                                 gamma_groups=gamma_groups_base * g_scale,
                                 dose_cap_total=Q_target)
            deaths[sname] = res['total_deaths']
            doses[sname] = res['Q_cumulative']
        order = ' < '.join(sorted(deaths, key=deaths.get))
        print(f"  порядок по смертности: {order}")
        row = {'perturbation': pname, **deaths,
               **{'Q_' + n: doses[n] for n in names}, 'order': order}
        save_row(path, fieldnames, row, first=(k == 0))


if __name__ == '__main__':
    DT = 0.125               # шаг по времени (для сравнения при dt = 0.5 поставьте 0.5)
    MESH_N = 50              # сетка 50x50, h = 2 км
    OUT = 'results_dt0125'   # папка для результатов (для dt = 0.5 например 'results_dt05')
    RUN_CONVERGENCE = True   # при dt = 0.5 сходимость повторять не нужно
    RUN_DOSE = True
    RUN_SENSITIVITY = True

    os.makedirs(OUT, exist_ok=True)
    print(f"Параметры запуска: mesh_n={MESH_N}, dt={DT}, папка={OUT}")
    if RUN_CONVERGENCE:
        convergence_study(output_dir=OUT)
    if RUN_DOSE:
        dose_equalized_strategies(mesh_n=MESH_N, dt=DT, output_dir=OUT)
    if RUN_SENSITIVITY:
        sensitivity_analysis(mesh_n=MESH_N, dt=DT, output_dir=OUT)
