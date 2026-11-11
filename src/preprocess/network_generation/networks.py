####################################################################################################################################
##
# Python script to generate the user-subreddit files from raw data and construct the networks
##
####################################################################################################################################

import os
import time
import config
import pandas as pd
import graph_tool.all as gt

from pickle import dump
from collections import defaultdict

def charge_month(year, month, dir=config.raw_comments, out_dir=config.comments, threshold=2):

    '''
        Process and save .parquet files for comments and submissions
        with a minimum threshold of user activity in a subreddit

        Input
            - year(int): year to process the data from
            - month(int): month to process the data from
            - dir(str, optional): Directory to read the data
            - threshold(int, optional): Minimum number of posts/comments 
            a user must have in a subreddit to be included. Default is 2.

        Output
            - None: Saves the data to a predefined directory in .pkl format
    '''
    
    all_data = []
    sub2users = defaultdict(set)
    suffix = f'_{threshold}' if threshold != 0 else '' # Assign suffix based on threshold value
    
    direct = dir
        
    temp_dir = f'{direct}year={year}/month={month}'
    lit = 'RC' if direct == config.raw_comments else 'RS'
    n_files = len(os.listdir(temp_dir))
        
    for i in range(n_files):
        try:
            all_data.append(
                pd.read_parquet(f'{temp_dir}/{lit}_{year}-{month:02d}-{i}.parquet', 
                                columns=['author', 'subreddit']))
        except Exception:
            continue

    if not all_data:
        return None

    known_bots = ['automoderator', '[deleted]']
    # Defining a regex pattern to identify likely bot usernames
    # notably, we could be over-restricting for those who end 
    # with 'bot' but are not bots
    bot_pattern = r'(?:bot$|^bot[_-]|[_-]bot[_-])'
    
    all_data = pd.concat(all_data, ignore_index=True)
    all_data = all_data.dropna(subset=['author', 'subreddit'])
    all_data = all_data[~all_data['subreddit'].str.startswith('u_')] #Remove user personal pages

    a = all_data['author'].str.lower()
    all_data = all_data[~a.str.contains(bot_pattern, regex=True) & ~a.isin(known_bots)] #Remove likely bots with regex

    counts = all_data.groupby(['subreddit', 'author'], observed=True).size()
    counts = counts[counts >= threshold].reset_index()
    
    for subreddit, users in counts.groupby('subreddit', observed=True)['author']:
        sub2users[subreddit].update(users)

    filtered = '' if suffix == '' else 'filtered'
    out_dir = f'{out_dir}/{filtered}/{year}'
    os.makedirs(out_dir, exist_ok=True)

    with open(f'{out_dir}/{month}{suffix}.pkl', 'wb') as out: dump(sub2users, out)

def charge_month_cs(year, month, out_dir=config.combined_path, threshold=2):

    '''
        Process and save .parquet files for comments and submissions
        with a minimum threshold of user activity in a subreddit

        Input
            - year(int): year to process the data from
            - month(int): month to process the data from
            - dir(str, optional): Directory to read the data
            - threshold(int, optional): Minimum number of posts/comments 
            a user must have in a subreddit to be included. Default is 2.

        Output
            - None: Saves the data to a predefined directory in .pkl format
    '''
    
    all_data = []
    sub2users = defaultdict(set)
    suffix = f'_{threshold}' if threshold != 0 else '' # Assign suffix based on threshold value
    
    for direct in [config.raw_comments, config.raw_submissions]:
        
        temp_dir = f'{direct}year={year}/month={month}'
        lit = 'RC' if direct == config.raw_comments else 'RS'
        n_files = len(os.listdir(temp_dir))
        
        if n_files == 0:
            continue
            
        for i in range(n_files):
            try:
                all_data.append(
                    pd.read_parquet(f'{temp_dir}/{lit}_{year}-{month:02d}-{i}.parquet', 
                                    columns=['author', 'subreddit']))
            except Exception:
                continue

    if not all_data:
        return None

    known_bots = ['automoderator', '[deleted]']
    # Defining a regex pattern to identify likely bot usernames
    # notably, we could be over-restricting for those who end 
    # with 'bot' but are not bots
    bot_pattern = r'(?:bot$|^bot[_-]|[_-]bot[_-])'
    
    all_data = pd.concat(all_data, ignore_index=True)
    all_data = all_data.dropna(subset=['author', 'subreddit'])
    all_data = all_data[~all_data['subreddit'].str.startswith('u_')] #Remove user personal pages

    a = all_data['author'].str.lower()
    all_data = all_data[~a.str.contains(bot_pattern, regex=True) & ~a.isin(known_bots)] #Remove likely bots with regex

    counts = all_data.groupby(['subreddit', 'author'], observed=True).size()
    counts = counts[counts >= threshold].reset_index()
    
    for subreddit, users in counts.groupby('subreddit', observed=True)['author']:
        sub2users[subreddit].update(users)

    filtered = '' if suffix == '' else 'filtered'
    out_dir = f'{out_dir}/{filtered}/{year}'
    os.makedirs(out_dir, exist_ok=True)

    with open(f'{out_dir}/{month}{suffix}.pkl', 'wb') as out: dump(sub2users, out)
    
def process_years(start=2005, end=2022, m_start=1, m_end=12, threshold=2):

    '''
        Function to process the raw data for all years and months
    
    '''

    for year in range(start,end+1):
        for month in range(m_start,m_end+1):
            charge_month(year, month, dir=config.raw_comments, out_dir=config.comments_path, threshold=threshold)
            charge_month(year, month, dir=config.raw_submissions, out_dir=config.submissions_path, threshold=threshold)
            charge_month_cs(year, month, threshold=threshold)
    
def load_year(year, name, path):
    '''
        Load the subreddit data for a given year

        Input
            - year(int): Year to load the data from
            - name(str): Name of the file to load ('_2', etc.)
            - path(str): Directory to read the data from.

        Return
            - total (defaultdict(set)): Dictionary with subreddit-user key-value
    '''

    # defaultdict is easier to update than a normal dict, as it initializes empty sets for new keys automatically
    total = defaultdict(set) 

    for month in range(1,13):
        try:
            sub2users = pd.read_pickle(f'{path}{year}/{month}{name}.pkl')
        except:
            continue
        for key, value in sub2users.items():
            total[key].update(value)

    print('Data loaded')
    
    return total
    
def generate_network(year, name, path):

    '''
        Create the network object for a given year

        Input
            - year(int): Year to create the network from
            - name(str): Name of the file to load (e.g., '_monthly', '_2', etc.)

        Return
            - g (graph_tool graph): Subreddit network
    '''
    
    data = load_year(year, name, path)
    
    sub2users = {}

    # Order the data in descending order. This will allow to empty the dictionary
    # and ease memory use
    for key in sorted(data, key=lambda k: len(data[k]), reverse=True):
        sub2users[key] = data[key]
        del(data[key])
        
    print('Communities ordered')
    
    sizes = [len(sub2users[key]) for key in sub2users.keys()]

    #Generate graph initializing vertex and network properties
    g = gt.Graph(len(sub2users), directed=False)
    g.ep['weight'] = g.new_edge_property('int')
    g.vp['size'] = g.new_vertex_property('int', vals=sizes)
    g.vp['name'] = g.new_vertex_property('string', vals=sub2users.keys())
    
    sub2items = list(sub2users.items())
    n_items = len(sub2items)

    #Delete unused variables
    del(sizes)
    del(sub2users)
    
    for i in range(n_items):
        
        _, users1 = sub2items[i]
        
        for j in range(i+1, n_items):
            
            _, users2 = sub2items[j]
            common = len(users1.intersection(users2))

            #If they share users, add edge and weight (number of users)
            if common!=0:
                e = g.add_edge(i,j)
                g.ep['weight'][e] = common

        #Empty used subreddits to ease memory
        sub2items[i] = (None, None)
        
    return g
    
def get_networks(name, data_path, fname='', start=2006, end=2022, networks_path=config.cs_f_networks_path):

    '''
        Call function for generate_network (all years)

        Input
            - name(str): Name of the file to load
            - fname(str, optional): Name of the file to save
            - start(int, optional): Start year. Default is 2006.
            - end(int, optional): End year. Default is 2022.
            - networks_path(str, optional): Directory to save the networks. Default is config.cs_f_networks_path.

        Output
            Saves the network in its respective directory.
    '''
    
    for year in range(start,end+1):

        print(f'Processing network for year {year}...')
        time0 = time.time()
        network = generate_network(year, name=name, path=data_path)
        network.save(f'{networks_path}{year}{name}{fname}.gt.gz')

        network = None
        
        print(f'Generated network for year {year}, elapsed time {time.time()-time0}')
        
def generate_all(threshold=2, start=2005, end=2022, path=config.combined_f_path):

    '''
        Call function to generate all the necessary 
        preprocessing and networks for all years and specifications
        
        Input
    '''


    # Thanks to modularity this function can be parallelized if
    # there's enough computational power to handle different newtorks
    # at once
    name = f'_{threshold}' if threshold != 0 else ''
    if name == '' and ('filtered' in path):
        print('Path for filtered files and no threshold selected')
        return None
    elif name != '' and ('filtered' not in path):
        print('Path for unfiltered files but threshold selected')
        return None
    process_years(start=start, end=end, threshold=threshold)
    get_networks(name, path, start=start, end=end)
