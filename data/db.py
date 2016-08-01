from config import config as cc
import mysql.connector
import pandas as pd

def getData():
    print('Downloading data...')

    con = mysql.connector.connect(
        user = cc.config['db']['user'],
        password = cc.config['db']['pass'],
        host = cc.config['db']['host'],
        database= cc.config['db']['name'])

    query = 'SELECT {} FROM {}'.format(
        ','.join(cc.fetch['cols']),
        cc.fetch['table'])

    if cc.fetch['where']:
        query += ' WHERE {}'.format(cc.fetch['where'])
    if cc.fetch['limit']:
        query += ' LIMIT {}'.format(cc.fetch['limit'])

    print(query)

    df = pd.read_sql(
        sql = query,
        con = con,
        index_col = cc.fetch['index_col'])

    con.close()

    print('...done')
    return df