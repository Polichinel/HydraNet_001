import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
import urllib.request
import os


def get_gwno():

    location = '/home/projects/ku_00017/data/raw/PRIO'
    #location = '/home/simon/Documents/Bodies/data/PRIO'#local
    #path_gwno = location + '/PRIO-GRID Yearly Variables for 2003-2009 - 2022-06-16.csv' #https://grid.prio.org/#/download # need to figrue out the API
    path_gwno = location + '/PRIO-GRID Yearly Variables for 1989-2014 - 2022-06-16.csv' #https://grid.prio.org/#/download # need to figrue out the API

    # why not just go 1989 - 2019 like ucdp...

    gwno = pd.read_csv(path_gwno)

    return gwno

def get_prio_shape():

    location = '/home/projects/ku_00017/data/raw/PRIO'
    #location = '/home/simon/Documents/Bodies/data/PRIO'#local
    path_prio = location + '/priogrid_shapefiles.zip'

    if os.path.isfile(path_prio) == True:
        
        print('File already downloaded')
        prio_grid = gpd.read_file('zip://' + path_prio)

        prio_grid =  pd.DataFrame(prio_grid.drop(columns = ['geometry']))

    else:
        print('Beginning file download PRIO...')
        url_prio = 'http://file.prio.no/ReplicationData/PRIO-GRID/priogrid_shapefiles.zip'

        urllib.request.urlretrieve(url_prio, path_prio)
        prio_grid = gpd.read_file('zip://' + path_prio)

        prio_grid =  pd.DataFrame(prio_grid.drop(columns = ['geometry']))

    return prio_grid


def get_gwno():

    location = '/home/projects/ku_00017/data/raw/PRIO'
    #location = '/home/simon/Documents/Bodies/data/PRIO' #local
    #path_gwno = location + '/PRIO-GRID Yearly Variables for 2003-2009 - 2022-06-16.csv' #https://grid.prio.org/#/download # need to figrue out the API
    path_gwno = location + '/PRIO-GRID Yearly Variables for 1989-2014 - 2022-06-16.csv' #https://grid.prio.org/#/download # need to figrue out the API

    # why not just go 1989 - 2019 like ucdp...

    gwno = pd.read_csv(path_gwno)

    return gwno


def get_ucdp():

    location = '/home/projects/ku_00017/data/raw/UCDP'
    #location = '/home/simon/Documents/Bodies/data/UCDP' #local
    path_ucdp = location + "/ged201-csv.zip"
    
    if os.path.isfile(path_ucdp) == True:
        print('file already downloaded')
        ucdp = pd.read_csv(path_ucdp, low_memory=False)


    else: 
        print('Beginning file download UCDP...')

        url_ucdp = 'https://ucdp.uu.se/downloads/ged/ged201-csv.zip'
    
        urllib.request.urlretrieve(url_ucdp, path_ucdp)
        ucdp = pd.read_csv(path_ucdp, low_memory=False)


    # just to save ram for now !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sub_years = sorted(ucdp['year'].unique())[:4]
    ucdp = ucdp[ucdp['year'].isin(sub_years)]
    # -----------------------------------------------------------------------------------------------------

    ucdp['month'] = pd.to_datetime(ucdp['date_start']).dt.month
    ucdp.rename(columns= {'priogrid_gid' : 'gid'}, inplace= True)

    # ONLY STATE BASED!
    ucdp = ucdp[ucdp['type_of_violence'] == 2].copy()

    feature_list = ['month','year', 'gid', 'deaths_a','deaths_b', 'deaths_civilians', 'deaths_unknown','best', 'high', 'low']

    ucdp_monthly_unit = ucdp.loc[:,feature_list].groupby(['month','year', 'gid']).sum().reset_index()
    ucdp_monthly_unit['log_best'] = np.log(ucdp_monthly_unit['best'] +1)
    ucdp_monthly_unit['log_low'] = np.log(ucdp_monthly_unit['low'] +1)
    ucdp_monthly_unit['log_high'] = np.log(ucdp_monthly_unit['high'] +1)

    ucdp_monthly_unit['in_ucdp'] = True # handy later when I wnat to remove water and stuff.

    return ucdp_monthly_unit




def get_views_data():

    #path_views = '/home/simon/Documents/Articles/ConflictNet/data/raw/ucdp_views_priogrid_month.csv.zip'
    path_views = '/home/projects/ku_00017/data/raw/conflictNet/ucdp_views_priogrid_month.csv.zip'

    if os.path.isfile(path_views) == True:

        df_views = pd.read_csv(path_views)

    else: 
        print('Beginning file download ViEWS...')
        
        url_views = 'https://views.pcr.uu.se/download/datasets/ucdp_views_priogrid_month.csv.zip'
        urllib.request.urlretrieve(url_views, path_views)
        df_views = pd.read_csv(path_views)

    df_views.rename(columns= {'pg_id':'gid'}, inplace = True)

    to_drop = ['id','ged_dummy_sb', 'ged_count_sb', 'ged_dummy_ns', 'ged_count_ns', 
            'ged_best_ns', 'ged_dummy_os', 'ged_count_os', 'ged_best_os']

    df_views.drop(columns=to_drop, inplace = True)


    # just to save ram on local !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # sub_years = sorted(df_views['year'].unique())[:4]
    # df_views = df_views[df_views['year'].isin(sub_years)]
    # -----------------------------------------------------------------------------------------------------

    return df_views


def monthly_grid(prio_grid, yearly_df):

    years = [sorted(yearly_df['year'].unique())] * prio_grid.shape[0]
    months = [list(np.arange(1, 13))] * prio_grid.shape[0]

    prio_grid['year'] = years
    prio_grid['month'] = months

    prio_grid = prio_grid.explode('year').reset_index(drop=True) 
    prio_grid = prio_grid.explode('month').reset_index(drop=True) 

    prio_grid['year'] = prio_grid['year'].astype(int)
    prio_grid['month'] = prio_grid['month'].astype(int)

    return prio_grid


def merge_grid_views(prio_grid, df_views):
    
    merge_on = list(set.intersection(set(prio_grid.columns.unique()),set(df_views.columns.unique())))
    merged_df = pd.merge(prio_grid, df_views, how = 'left', on = merge_on)

    merged_df.fillna({'ged_best_sb' : 0, 'gwcode' : 0}, inplace = True) # for gwno 0 is no country
    merged_df.fillna({'in_ucdp' : False}, inplace = True)

    merged_df['log_best'] = np.log(merged_df['ged_best_sb'] + 1)

    return merged_df


def add_month_id(prio_grid):

    prio_grid['year_month'] = prio_grid['year'].astype(str) + '_' + prio_grid['month'].astype(str) 

    ts = prio_grid['year_month'].unique()
    n_ts = len(ts)
    month_ids = np.arange(109, n_ts + 109, 1)
    month_id_df = pd.DataFrame({'year_month' : ts, 'month_id': month_ids})
    prio_grid = prio_grid.merge(month_id_df, on = 'year_month', how = 'left')

    prio_grid.drop(columns=['year_month'], inplace= True)

    return prio_grid


def get_views_sub(merged_df, df_views):

    africa_gids = df_views['gid'].unique()

    max_coords = merged_df[merged_df['gid'].isin(africa_gids)][['xcoord', 'ycoord']].max() + (1,3.25) # get both dim to 180
    min_coords = merged_df[merged_df['gid'].isin(africa_gids)][['xcoord', 'ycoord']].min() - (1,3)

    mask1 = ((merged_df['xcoord'] < max_coords[0]) & (merged_df['xcoord'] > min_coords[0]) & (merged_df['ycoord'] < max_coords[1]) & (merged_df['ycoord'] > min_coords[1]))
    views_subset = merged_df[mask1].copy()
    col_to_change = views_subset.columns[7:18]
    #views_subset.loc[~views_subset['gid'].isin(africa_gids), col_to_change] = 0
    views_subset.reset_index(inplace=True, drop = True)

    return views_subset



# this

def make_volumn(df):

    # we start with wat we know - but there is no reason not to try with more down til line.

    #sub_df = df[['gid', 'xcoord', 'ycoord', 'month_id', 'best', 'low', 'high', 'log_best', 'log_low', 'log_high', 'gwno']].copy() # remove the everything also the geo col.
    sub_df = df[['gid', 'xcoord', 'ycoord', 'month_id', 'ged_best_sb', 'log_best', 'gwcode']].copy() # remove the everything also the geo col.

    sub_df_sorted = sub_df.sort_values(['month_id', 'ycoord', 'xcoord'], ascending = [True, False, True])

    # try to keep the jazz
    #grid_ucdpS = grid_ucdpS[['gid','best', 'low',  'high', 'log_best', 'log_low', 'log_high']].copy() # remove the everything also the geo col. But keep gid. Why not.

    x_dim = sub_df['xcoord'].unique().shape[0]
    y_dim = sub_df['ycoord'].unique().shape[0]
    z_dim = sub_df['month_id'].unique().shape[0]

    ucpd_vol = np.array(sub_df_sorted).reshape((z_dim, y_dim, x_dim, -1))

    return ucpd_vol


def compile():

    prio_grid = get_prio_shape()
    gwno = get_gwno()
    ucdp = get_ucdp()
    df_views = get_views_data()
    prio_grid = monthly_grid(prio_grid, df_views)
    prio_grid = add_month_id(prio_grid)
    merged_df = merge_grid_views(prio_grid, df_views)
    views_subset = get_views_sub(merged_df, df_views)

    views_vol = make_volumn(views_subset)
    print(views_vol.shape)

    worlds_vol = make_volumn(merged_df)
    print(worlds_vol.shape)

    # Pickle
    location = '/home/projects/ku_00017/data/raw/conflictNet'
    #location = '/home/simon/Documents/Articles/ConflictNet/data/raw'


    print('Saving pickle1')
    file_name = "/views_monthly_REP_vol.pkl"
    output = open(location + file_name, 'wb')
    pickle.dump(views_vol, output)
    output.close()
    print(f'Pickled {file_name}')


    # NOT REALLY THE WOLRD RIGHT NOW!!! JUST THE FULL GRID!
    print('Saving pickle2')
    file_name = "/views_world_REP_monthly_vol.pkl"
    output = open(location + file_name, 'wb')
    pickle.dump(worlds_vol, output)
    output.close()
    print(f'Pickled {file_name}')

    print('Done')

if __name__ == '__main__':
    compile()

