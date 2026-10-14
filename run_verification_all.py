import csv
import numpy as np
from scipy.stats import pearsonr, spearmanr

FILES = ['real_data_omicron2022.csv', 'real_data_delta2021.csv']
DEFS = [('cases', 'B: подтв.-выздор.-умерш.'), ('cases_A', 'A: сумма новых за 14 сут')]
MESH_N, DT = 50, 0.125
MAX_LAG = 90

def load_columns(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    days = np.array([float(r['day']) for r in rows])
    cols = {c: np.array([float(r[c]) for r in rows]) for c in rows[0] if c in ('cases', 'cases_A')}
    return days, cols

def best_shift(model, real, max_lag):
    best = (-2.0, 0)
    n = len(real)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = model[:n - lag], real[lag:]
        else:
            a, b = model[-lag:], real[:n + lag]
        if len(a) < 30:
            continue
        r = pearsonr(a, b)[0]
        if r > best[0]:
            best = (r, lag)
    return best


def main(model_days=None, model_I=None):
    if model_days is None:
        from model_core import run_simulation, VACC_STRATEGY_UNIFORM
        res = run_simulation(mesh_n=MESH_N, dt=DT, upsilon_groups=VACC_STRATEGY_UNIFORM)
        model_days = np.asarray(res['time_points']) + DT       # состояние после шага
        model_I = np.asarray(res['I_total_series'])
    print(f"Модель: пик I = {model_I.max():.0f} на сутки {model_days[model_I.argmax()]:.1f}")

    summary = [['file', 'definition', 'n', 'pearson_r', 'pearson_p', 'spearman_rho',
                'real_peak_day', 'real_peak', 'model_peak_day', 'model_peak',
                'best_shift_days', 'r_at_best_shift']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for fname in FILES:
        try:
            days, cols = load_columns(fname)
        except FileNotFoundError:
            print(f"Файл {fname} не найден, пропускаю")
            continue
        m = np.interp(days, model_days, model_I)
        fig, axes = plt.subplots(1, len(DEFS), figsize=(6.5 * len(DEFS), 4.3))
        for ax, (col, label) in zip(np.atleast_1d(axes), DEFS):
            if col not in cols:
                continue
            real = cols[col]
            r, p = pearsonr(m, real)
            rho = spearmanr(m, real)[0]
            rb, lag = best_shift(m, real, MAX_LAG)
            summary.append([fname, col, len(days), round(r, 3), p, round(rho, 3),
                            days[real.argmax()], real.max(), model_days[model_I.argmax()],
                            model_I.max(), lag, round(rb, 3)])
            print(f"{fname} [{col}]: r = {r:.3f} (p = {p:.3g}), Спирмен = {rho:.3f}; "
                  f"пик данных: сутки {days[real.argmax()]:.0f} ({real.max():.0f}), "
                  f"пик модели: сутки {model_days[model_I.argmax()]:.0f} ({model_I.max():.0f}); "
                  f"[диагностика] r при сдвиге {lag:+d} сут = {rb:.3f}")
            ax.plot(days, real, lw=2, label=f'данные ({label})')
            ax.plot(days, m, lw=2, ls='--', label='модель I(t), стратегия 1')
            ax.set_xlabel('сутки от t=0'); ax.set_ylabel('активные случаи / I')
            ax.set_title(f'{fname}: r = {r:.2f}', fontsize=10)
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig('verification_' + fname.replace('.csv', '') + '.png', dpi=150)
        plt.close(fig)

    with open('verification_summary.csv', 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(summary)
    print("Готово: verification_summary.csv и рисунки verification_*.png")


if __name__ == '__main__':
    main()
