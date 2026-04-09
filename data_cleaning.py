import pandas as pd
UNIVERSITY_ZIPS = {
    '02215',  # Boston University
    '02138',  # Harvard University
    '02139',  # MIT
    '02115',  # Northeastern, Wentworth, MassArt, New England Conservatory, Berklee
    '02125',  # UMass Boston
    '02155',  # Tufts University
    '02134',  # Harvard Business School
    '02163',  # Harvard Business School
    '02116',  # Emerson College
    '02120',  # Simmons University
    '02130',  # Emmanuel College
    '02114',  # Suffolk University
    '02129',  # Bunker Hill Community College
}

ZIP_TO_NEIGHBORHOOD = {
    # Allston/Brighton
    '02134': 'Allston/Brighton', '02135': 'Allston/Brighton', '02163': 'Allston/Brighton',
    # Back Bay/Beacon Hill
    '02108': 'Back Bay/Beacon Hill', '02116': 'Back Bay/Beacon Hill', '02117': 'Back Bay/Beacon Hill',
    '02123': 'Back Bay/Beacon Hill', '02133': 'Back Bay/Beacon Hill', '02199': 'Back Bay/Beacon Hill',
    '02216': 'Back Bay/Beacon Hill', '02217': 'Back Bay/Beacon Hill', '02295': 'Back Bay/Beacon Hill',
    # Central Boston
    '02101': 'Central Boston', '02102': 'Central Boston', '02103': 'Central Boston',
    '02104': 'Central Boston', '02105': 'Central Boston', '02106': 'Central Boston',
    '02107': 'Central Boston', '02109': 'Central Boston', '02110': 'Central Boston',
    '02111': 'Central Boston', '02112': 'Central Boston', '02113': 'Central Boston',
    '02114': 'Central Boston', '02196': 'Central Boston', '02201': 'Central Boston',
    '02202': 'Central Boston', '02203': 'Central Boston', '02204': 'Central Boston',
    '02205': 'Central Boston', '02206': 'Central Boston', '02207': 'Central Boston',
    '02208': 'Central Boston', '02209': 'Central Boston', '02211': 'Central Boston',
    '02212': 'Central Boston', '02222': 'Central Boston', '02293': 'Central Boston',
    # Charlestown
    '02129': 'Charlestown',
    # Dorchester
    '02122': 'Dorchester', '02124': 'Dorchester', '02125': 'Dorchester',
    # East Boston
    '02128': 'East Boston', '02228': 'East Boston',
    # Fenway/Kenmore
    '02115': 'Fenway/Kenmore', '02215': 'Fenway/Kenmore',
    # Hyde Park
    '02136': 'Hyde Park',
    # Jamaica Plain
    '02130': 'Jamaica Plain',
    # Mattapan
    '02126': 'Mattapan',
    # Roslindale
    '02131': 'Roslindale',
    # Roxbury
    '02119': 'Roxbury', '02120': 'Roxbury', '02121': 'Roxbury',
    # South Boston
    '02127': 'South Boston', '02210': 'South Boston',
    # South End
    '02118': 'South End',
    # West Roxbury
    '02132': 'West Roxbury',
    # Malden
    '02148': 'Malden',
    # Medford
    '02155': 'Medford',
    # Revere
    '02151': 'Revere',
    # Quincy
    '02169': 'Quincy',
    # Cambridge
    '02139': 'Cambridge', '02138': 'Cambridge', '02141': 'Cambridge',
    # Brookline
    '02446': 'Brookline', '02445': 'Brookline', '02171': 'Brookline',
    # Arlington
    '02474': 'Arlington',
    # Somerville
    '02145': 'Somerville', '02144': 'Somerville', '02143': 'Somerville',
}

def has_university(zipcode):
    key = str(zipcode).strip().zfill(5)
    return 1 if key in UNIVERSITY_ZIPS else 0

def zip_to_neighborhood(zipcode, unknown_label='Unknown'):
    """Map a zip code to its Boston neighborhood.
    
    Args:
        zipcode: zip code as int or string (handles zero-padding automatically)
        unknown_label: value to return if zip code not found
    
    Returns:
        Neighborhood name string, or unknown_label if not found
    """
    key = str(zipcode).strip().zfill(5)
    return ZIP_TO_NEIGHBORHOOD.get(key, unknown_label)

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
    boston_df['Neighborhood'] = boston_df['ZipCode'].apply(zip_to_neighborhood)
    boston_df['HasUniversity'] = boston_df['ZipCode'].apply(has_university)
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
#reshape_to_long()