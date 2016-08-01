from config import config as cc
import mysql.connector
import pandas as pd

def getCon():
    con = mysql.connector.connect(
        user = cc.config['db']['user'],
        password = cc.config['db']['pass'],
        host = cc.config['db']['host'],
        database= cc.config['db']['name'])
    return con

def getData():
    print('Downloading data...')

    con = getCon()

    query = 'SELECT {} FROM {}'.format(
        ','.join(cc.exp['fetch']['cols']),
        cc.exp['fetch']['table'])

    if cc.exp['fetch']['where']:
        query += ' WHERE {}'.format(cc.exp['fetch']['where'])
    if cc.exp['fetch']['limit']:
        query += ' LIMIT {}'.format(cc.exp['fetch']['limit'])

    print(query)

    df = pd.read_sql(
        sql = query,
        con = con,
        index_col = cc.exp['fetch']['index_col'])

    con.close()

    print('...done')
    return df

def sendStatistics(**kwargs):
    print('Sending statistics...')

    con = getCon()

    cols = []
    vals = []

    for col,val in kwargs.iteritems():
        cols.append(col)
        vals.append(val)

    query = 'INSERT INTO {} ({}) VALUES ({})'.format(
        cc.config['statistics']['table'],
        ','.join(cols),
        ','.join(['%s']*len(cols)))1

    print(query)

    cursor = con.cursor()

    try:
        cursor.execute(query, tuple(vals))
        con.commit()
    except Exception as e:
        con.rollback()
        print 'Exception:', e

    cursor.close()
    con.close()

    print('...done')
