# Use viewser env

from viewser import Queryset, Column
from ingester3.extensions import *
#from ingester3.DBWriter import DBWriter

import urllib.request
import os
import sys

import pickle
import numpy as np
import pandas as pd
import geopandas as gpd


def get_views_date(location, partitioner_dict):

    path_views_data = location + '/views_data.pkl' # this does not work with your current naming...

    if os.path.isfile(path_views_data) == True:

        print('File already downloaded')
        df = pd.read_pickle(path_views_data)
        
    else:
        print('Beginning file download through viewser...')

        # queryset_base = (Queryset("simon_tests", "priogrid_month")
        #     .with_column(Column("sb_best_count_pgm", from_table = "ged2_pgm", from_column = "ged_sb_best_count_nokgi").transform.ops.ln().transform.missing.replace_na()))

        queryset_base = (Queryset("simon_tests", "priogrid_month")
            .with_column(Column("sb_best_count_pgm", from_table = "ged2_pgm", from_column = "ged_sb_best_count_nokgi").transform.ops.ln().transform.missing.replace_na())
            .with_column(Column("sb_high_count_pgm", from_table = "ged2_pgm", from_column = "ged_sb_high_count_nokgi").transform.ops.ln().transform.missing.replace_na())
            .with_column(Column("month", from_table = "month", from_column = "month"))
            .with_column(Column("year_id", from_table = "country_year", from_column = "year_id"))
            .with_column(Column("c_id", from_table = "country_year", from_column = "country_id")))


        # You want high, and you want ns and os.

        df = queryset_base.publish().fetch()

        df.reset_index(inplace = True)

        df.rename(columns={'priogrid_gid': 'pg_id', 'sb_best_count_pgm' : 'ln_sb_best', 'sb_high_count_pgm' : 'ln_sb_high'}, inplace= True)


        
        # df = df[df['month_id'] == 121] # temp sub
        # df = df[df['month_id'].isin([121, 122, 123, 124])] # temp sub
        
        
        month_range = np.arange(partitioner_dict['train'][0], partitioner_dict['predict'][1]+1,1)
        #month_range = np.arange(partitioner_dict['train'][0],partitioner_dict['train'][0]+11,1)
        #month_range = np.arange(partitioner_dict['train'][0],partitioner_dict['train'][0]+24,1)


        df = df[df['month_id'].isin(month_range)] # temp sub


        #df['lat'] = df.pg.lat # already in PRIO grid as ycoord
        #df['lon'] = df.pg.lon # already in PRIO grid as xcoord

        #df['month'] = df.pgm.month # See if these can be optained through a query_set
        #df['year_id'] = df.pgm.year # See if these can be optained through a query_set
        #df['row'] = df.pgm.row # already in PRIO grid as row
        #df['col'] = df.pgm.col # already in PRIO grid as col

        #df['c_id'] = df.pgy.c_id # See if these can be optained through a query_set

        df['in_viewser'] = True
        # df['name'] = df.cy.name # No need        
        # df.to_pickle(path_views_data)
        # print('VIEWS data pickled.')

    return df


def get_prio_shape(location):

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

    prio_grid.rename(columns={'gid': 'pg_id'}, inplace= True)

    return prio_grid


def monthly_grid(prio_grid, views_df):

    years = [sorted(views_df['year_id'].unique())] * prio_grid.shape[0]
    #months = [list(np.arange(1, 13))] * prio_grid.shape[0]

    months = [sorted(views_df['month'].unique())] * prio_grid.shape[0] # then you only get one for the test runs# expensive to get these

    prio_grid['year_id'] = years
    prio_grid['month'] = months

    prio_grid = prio_grid.explode('year_id').reset_index(drop=True) 
    prio_grid = prio_grid.explode('month').reset_index(drop=True) 

    prio_grid['year_id'] = prio_grid['year_id'].astype(int)
    prio_grid['month'] = prio_grid['month'].astype(int)

    # # Add month_id:
    # prio_grid['year_month'] = prio_grid['year'].astype(str) + '_' + prio_grid['month'].astype(str)
    # month_ids = np.arange(views_df['month_id'].min(), views_df['month_id'].max()+1, 1)
    
    # # Hack, but it works
    # ts = prio_grid['year_month'].unique()
    # month_id_df = pd.DataFrame({'year_month' : ts, 'month_id': month_ids})
    # prio_grid = prio_grid.merge(month_id_df, on = 'year_month', how = 'left')

    # Merge
    # full_grid = prio_grid.merge(views_df, on = ['pg_id', 'year_id', 'month', 'col', 'row'], how = 'left')
    full_grid = prio_grid.merge(views_df, on = ['pg_id', 'year_id', 'month'], how = 'left')

    full_grid.fillna({'ln_sb_best' : 0, 'ln_sb_high' : 0, 'c_id' : 0, 'in_viewser' : False}, inplace = True) # for c_id 0 is no country

    # full_grid["month_id"] = full_grid.groupby("month").transform(lambda x: x.fillna(x.mean(skipna = True)))['month_id'] # I think this is cool, but must check...
    full_grid["month_id"] = full_grid.groupby(["year_id", "month"]).apply(lambda x: x.fillna(x.mean(skipna = True)))['month_id']

    # Drop stuff..
    full_grid.dropna(inplace=True)
    # the point of this is to drop months that were not give and month_id. The PRIO grid explosion makes only whole years, so this removes any excess months

    return full_grid



def get_sub_grid(grid, views_df):

        views_gids = views_df['pg_id'].unique()

        # get both dim to 180. Fine since you maxpool(2,2) two time: 180 -> 90 -> 45
        # A better number might be 192 since: 192 -> 96 -> 48 -> 24 -> 12 -> 6 -> 3
        max_coords = grid[grid['pg_id'].isin(views_gids)][['xcoord', 'ycoord']].max() + (1,1) 
        min_coords = grid[grid['pg_id'].isin(views_gids)][['xcoord', 'ycoord']].min() - (1,0.25) 
        
        # Maks it
        mask1 = ((grid['xcoord'] < max_coords[0]) & (grid['xcoord'] > min_coords[0]) & (grid['ycoord'] < max_coords[1]) & (grid['ycoord'] > min_coords[1]))
        grid = grid[mask1].copy()

        return grid


def make_volumn(grid):

    # we start with wat we know - but there is no reason not to try with more down til line.

    sub_df = grid[['pg_id', 'xcoord', 'ycoord', 'month_id', 'c_id', 'ln_sb_best', 'ln_sb_high']].copy() # remove the everything also the geo col. What about in_viewser?

    sub_df_sorted = sub_df.sort_values(['month_id', 'ycoord', 'xcoord'], ascending = [True, False, True])

    # try to keep the jazz
    #grid_ucdpS = grid_ucdpS[['gid','best', 'low',  'high', 'log_best', 'log_low', 'log_high']].copy() # remove the everything also the geo col. But keep gid. Why not.

    x_dim = sub_df['xcoord'].unique().shape[0]
    y_dim = sub_df['ycoord'].unique().shape[0]
    z_dim = sub_df['month_id'].unique().shape[0]

    ucpd_vol = np.array(sub_df_sorted).reshape((z_dim, y_dim, x_dim, -1))

    return ucpd_vol


def compile():

    from pkg_resources import get_distribution
    installed_version = get_distribution('ingester3').version

    if installed_version >= '1.8.1':
        print (f"You are running version {installed_version} which is consistent with the documentation")
    else:
        print (f"""You are running an obsolete version ({installed_version}). Run: pip install ingester3 --upgrade to upgrade""")


    partitioner = input(f'a) Calibration \nb) Test \nc) Future\n')

    if partitioner == 'a':
        partitioner_dict = {"train":(121,396),"predict":(397,444)} # calib_partitioner_dict
        file_name = "/viewser_monthly_vol_calib.pkl"

    elif partitioner == 'b':
        partitioner_dict = {"train":(121,444),"predict":(445,492)} # test_partitioner_dict
        file_name = "/viewser_monthly_vol_test.pkl"

    elif partitioner == 'c':
        partitioner_dict = {"train":(121,492),"predict":(493,504)} # furture_partitioner_dict
        file_name = "/viewser_monthly_vol_furture.pkl"

    else:
        print('Wrong input... breaking operation.')
        sys.exit() # you could let people in they own stuff..


    #location = '/home/number_one/Documents/scripts/conflictNet/data/raw'
    location = '/home/projects/ku_00017/data/raw/conflictNet'

    df = get_views_date(location, partitioner_dict)
    print('Data loaded from viewser')
    
    grid = get_prio_shape(location)
    print('PRIO-grid loaded')
    
    grid = monthly_grid(grid, df)
    print('PRIO-grid and viewser data merged')
    
    grid = get_sub_grid(grid, df)
    print('Sub-grid partitioned')

    vol = make_volumn(grid)
    print('Created volumn')

    print(f'Pickling {file_name}')
    output = open(location + file_name, 'wb')
    pickle.dump(vol, output)
    output.close()
    print(f'Pickled {file_name}')
    print('Done.')


if __name__ == "__main__":
    compile()
