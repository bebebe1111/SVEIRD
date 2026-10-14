import csv
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')           
import matplotlib.pyplot as plt

MESH_N = 50                     
DT = 0.125
T = 160.0
OUT_DIR = 'figures'
K_CSV = os.path.join('results_dt0125', 'dose_equalized_strategies.csv')
FORCE_RECOMPUTE = False
SNAPSHOT_EVERY_DAYS = 1.0  
SHARED_COLOR_SCALE_FOR_AGE_MAPS = True 

STRATEGY_FILE = {'Равномерная': 's1', 'Приоритет пожилых': 's2', 'Приоритет молодых': 's3'}
AGE_LABELS = ['0–10', '11–18', '19–25', '26–44', '45–59', '60–74', '75+']
AGE_TITLES = ['Дети (0–10)', 'Подростки (11–18)', 'Молодёжь (19–25)',
              'Молодые взрослые (26–44)', 'Средний возраст (45–59)',
              'Пожилые (60–74)', 'Старческий возраст (75+)', 'Все группы (сумма)']
COMP_NAMES = ['S', 'V', 'E', 'I', 'R', 'D']


def load_k():
    if not os.path.exists(K_CSV):
        raise FileNotFoundError(f"Не найден {K_CSV}. Сначала запустите run_experiments.py "
                                f"(он считает множители k для равного числа доз).")
    k = {}
    with open(K_CSV, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            k[row['strategy']] = float(row['rate_scale_k'])
    missing = [name for name in STRATEGY_FILE if name not in k]
    if missing:
        raise ValueError(f"В {K_CSV} нет множителей k для стратегий: {missing}")
    print(f"k прочитаны из {K_CSV}: {k}")
    return k

def compute_data():
    from model_core import (run_simulation, VACC_STRATEGY_UNIFORM,
                            VACC_STRATEGY_ELDERLY_PRIORITY, VACC_STRATEGY_YOUTH_PRIORITY)
    base = {
        'Равномерная': VACC_STRATEGY_UNIFORM,
        'Приоритет пожилых': VACC_STRATEGY_ELDERLY_PRIORITY,
        'Приоритет молодых': VACC_STRATEGY_YOUTH_PRIORITY,
    }
    k = load_k()
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, ups in base.items():
        path = os.path.join(OUT_DIR, f'data_{STRATEGY_FILE[name]}.pkl')
        if os.path.exists(path) and not FORCE_RECOMPUTE:
            print(f"[{name}] данные уже есть ({path}), расчёт пропущен")
            continue
        print(f"[{name}] считаю: mesh_n={MESH_N}, dt={DT}, k={k[name]:.4f} ...")
        res = run_simulation(mesh_n=MESH_N, dt=DT, T=T,
                             upsilon_groups=ups * k[name],
                             record_series=True,
                             snapshot_every_days=SNAPSHOT_EVERY_DAYS)
        res['strategy'] = name
        res['k'] = k[name]
        with open(path, 'wb') as f:
            pickle.dump(res, f)
        print(f"[{name}] готово: смертей {res['total_deaths']:.0f}, доз Q = {res['Q_cumulative']:.0f}")


def load_data(name):
    with open(os.path.join(OUT_DIR, f'data_{STRATEGY_FILE[name]}.pkl'), 'rb') as f:
        return pickle.load(f)

def r0_ngm():
    from model_core import (beta_groups_base, gamma_groups_base, alpha_groups,
                            population_distribution, contact_matrix)
    K = ((population_distribution * beta_groups_base)[:, None] * contact_matrix
         / (gamma_groups_base + alpha_groups)[None, :])
    return float(max(abs(np.linalg.eigvals(K))))


def field_on_grid(xy, values):
    n = int(round(np.sqrt(len(values))))
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    return values[order].reshape(n, n)

def nearest_snapshot(data, day):
    idx = int(np.argmin(np.abs(data['snap_times'] - day)))
    return idx, float(data['snap_times'][idx])

def peak_day(data):
    I_tot = data['series_totals'][:, 3, :].sum(axis=1)
    return float(data['series_times'][int(np.argmax(I_tot))])

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("сохранено:", path)

def plot_total_dynamics(data, filename):
    t = data['series_times']
    tot = data['series_totals'].sum(axis=2)      
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'black']
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, name, col in zip(range(6), COMP_NAMES, colors):
        ax.plot(t, tot[:, c], label=name, color=col, lw=1.8)
    ax.set_xlabel('t, сут')
    ax.set_ylabel('Численность, чел.')
    ax.set_title(f"Суммарная динамика компартментов — {data['strategy'].lower()} вакцинация\n"
                 f"$R_0$ = {r0_ngm():.2f} (метод матрицы следующего поколения)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3)
    ax.set_xlim(0, t[-1])
    save(fig, filename)

def plot_age_I(data, filename, ymax):
    t = data['series_times']
    I = data['series_totals'][:, 3, :]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for g in range(7):
        ax.plot(t, I[:, g], label=AGE_LABELS[g], lw=1.6)
    ax.set_xlabel('t, сут')
    ax.set_ylabel('Заразные I, чел.')
    ax.set_title(f"I(t) по возрастным группам — {data['strategy'].lower()}")
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, t[-1])
    ax.grid(alpha=0.3)
    ax.legend(title='Возраст', fontsize=8)
    save(fig, filename)

def plot_snapshots(data, filename, days=(0, 40, 80, 120, 160)):
    xy = data['dof_xy']
    maps, labels = [], []
    for d in days:
        idx, t_real = nearest_snapshot(data, d)
        maps.append(field_on_grid(xy, data['snap_I'][idx].sum(axis=0)))
        labels.append(f"День {int(round(t_real))}")
    vmax = max(m.max() for m in maps)
    n = len(days)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), squeeze=False)
    axes = axes[0]
    for ax, m, lab in zip(axes, maps, labels):
        im = ax.imshow(m, origin='lower', extent=[0, 100, 0, 100],
                       cmap='inferno', vmin=0, vmax=vmax)
        ax.set_title(lab)
        ax.set_xlabel('x, км')
    axes[0].set_ylabel('y, км')
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
    cbar.set_label('I (эквивалентная численность)')
    fig.suptitle('Пространственное распределение I — все возрасты (суммарно), '
                 f"{data['strategy'].lower()} вакцинация")
    save(fig, filename)

def plot_age_peak(data, filename, vmax_per_panel=None):
    xy = data['dof_xy']
    day = peak_day(data)
    idx, t_real = nearest_snapshot(data, day)
    fields = [field_on_grid(xy, data['snap_I'][idx][g]) for g in range(7)]
    fields.append(sum(fields))
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for p, (ax, f) in enumerate(zip(axes.ravel(), fields)):
        vmax = vmax_per_panel[p] if vmax_per_panel is not None else f.max()
        im = ax.imshow(f, origin='lower', extent=[0, 100, 0, 100],
                       cmap='plasma', vmin=0, vmax=vmax)
        ax.set_title(AGE_TITLES[p], fontsize=11)
        ax.set_xlabel('x, км'); ax.set_ylabel('y, км')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Пространственное распределение I по возрастным группам — день {int(round(t_real))} "
                 f"(пик), {data['strategy'].lower()} вакцинация", fontsize=14)
    fig.tight_layout()
    save(fig, filename)
    return t_real

def age_peak_fields(data):
    xy = data['dof_xy']
    idx, _ = nearest_snapshot(data, peak_day(data))
    f = [field_on_grid(xy, data['snap_I'][idx][g]) for g in range(7)]
    f.append(sum(f))
    return f

def make_plots():
    os.makedirs(OUT_DIR, exist_ok=True)
    d1 = load_data('Равномерная')
    d2 = load_data('Приоритет пожилых')

    plot_total_dynamics(d1, 'Kychkina_ris1.png')

    ymax = 1.05 * max(d1['series_totals'][:, 3, :].max(), d2['series_totals'][:, 3, :].max())
    plot_age_I(d1, 'Kychkina_ris2_1.png', ymax)  
    plot_age_I(d2, 'Kychkina_ris2_2.png', ymax)

    plot_snapshots(d1, 'Kychkina_ris3.png')

    vmax = None
    if SHARED_COLOR_SCALE_FOR_AGE_MAPS:
        f1, f2 = age_peak_fields(d1), age_peak_fields(d2)
        vmax = [max(a.max(), b.max()) for a, b in zip(f1, f2)]
    day1 = plot_age_peak(d1, 'Kychkina_ris4_1.png', vmax)
    day2 = plot_age_peak(d2, 'Kychkina_ris4_2.png', vmax)
    print(f"Дни пика: стратегия 1 — {day1:.1f}, стратегия 2 — {day2:.1f} "
          f"(запишите их в подписи к рис. 3)")

    for d in (d1, d2):
        I_peak_by_group = d['series_totals'][:, 3, :].max(axis=0)
        print(f"{d['strategy']}: смертей {d['total_deaths']:.0f}, Q = {d['Q_cumulative']:.0f}, "
              f"пики I по группам {np.round(I_peak_by_group).astype(int)}")
        if 'D_by_group' in d:
            print("   умершие по группам:", np.round(d['D_by_group']).astype(int))

if __name__ == '__main__':
    compute_data()
    make_plots()
