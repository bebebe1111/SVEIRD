import pandas as pd

XLSX = 'data.xlsx'
REGION = 'Республика Саха (Якутия)'
ACTIVE_DAYS = 14
HORIZON = 160         

WAVES = [
    ('real_data_delta2021.csv',   '2021-07-15'),
    ('real_data_omicron2022.csv', '2022-01-10'),
]

def load_series():
    x = pd.ExcelFile(XLSX, engine='openpyxl')
    out = {}
    for sheet in ('CONFIRMED', 'RECOVERED', 'DEATHS'):
        df = x.parse(sheet)
        row = df[df['REGION'] == REGION].iloc[0, 1:]
        s = pd.to_numeric(row, errors='coerce').fillna(0)
        s.index = pd.to_datetime(s.index).normalize()
        out[sheet] = s[~s.index.duplicated(keep='first')] 
    d = pd.DataFrame(out)
    d['active_A'] = d['CONFIRMED'].rolling(ACTIVE_DAYS, min_periods=1).sum()
    d['active_B'] = d['CONFIRMED'].cumsum() - d['RECOVERED'].cumsum() - d['DEATHS'].cumsum()
    return d

def main():
    d = load_series()
    print(f"Загружено {len(d)} суток: {d.index[0].date()} ... {d.index[-1].date()}")
    for fname, t0 in WAVES:
        t0 = pd.Timestamp(t0)
        w = d.loc[t0: t0 + pd.Timedelta(days=HORIZON)].copy()
        out = pd.DataFrame({
            'day': (w.index - t0).days,
            'cases': w['active_B'].round(0).astype(int).values,
            'date': w.index.strftime('%Y-%m-%d'),
            'cases_A': w['active_A'].round(0).astype(int).values,
        })
        out.to_csv(fname, index=False, encoding='utf-8')
        print(f"{fname}: {len(out)} точек, максимум cases = {out['cases'].max()} "
              f"на день {int(out.loc[out['cases'].idxmax(), 'day'])}")

if __name__ == '__main__':
    main()
