import pandas as pd

def clean_zori_data():
    df = pd.read_csv('ZORI_monthly_by_ZIP.csv')
    df.rename(columns={'RegionName': 'ZipCode'}, inplace=True)
    df['ZipCode'] = df['ZipCode'].astype(str).str.zfill(5)  
    boston_df = df[df['Metro'].str.contains('Boston', case=False, na=False)].copy()
    boston_df.drop_duplicates(inplace=True)

    meta_cols = ['RegionID', 'SizeRank', 'ZipCode', 'RegionType',
                 'StateName', 'State', 'City', 'Metro', 'CountyName']
    date_cols = [c for c in boston_df.columns if c not in meta_cols]
    boston_df.dropna(subset=date_cols, how='all', inplace=True)

    boston_df.dropna(subset=date_cols, thresh=int(len(date_cols) * 0.8), inplace=True)
    boston_df[date_cols] = boston_df[date_cols].interpolate(axis=1, limit_direction='both')

    critical_meta = ['RegionID', 'ZipCode', 'State', 'City']
    boston_df.dropna(subset=critical_meta, inplace=True)
    boston_df.reset_index(drop=True, inplace=True)

    boston_df.to_csv('boston_cleaned_data.csv', index = False)


def reshape_to_long(wide_df):
    meta_cols = ['RegionID', 'SizeRank', 'ZipCode', 'RegionType',
                 'StateName', 'State', 'City', 'Metro', 'CountyName']
    date_cols = [c for c in wide_df.columns if c not in meta_cols]
 
    long_df = wide_df.melt(
        id_vars=meta_cols,
        value_vars=date_cols,
        var_name='date',
        value_name='zori'
    )
    long_df['date'] = pd.to_datetime(long_df['date'])
    long_df = long_df.sort_values(['ZipCode', 'date']).reset_index(drop=True)
    return long_df

clean_zori_data()
reshape_to_long()