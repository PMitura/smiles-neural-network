from config import config as cc
import psycopg2
import pandas as pd
import numpy


def getCon():
    # try:
    con = psycopg2.connect(
            user = cc.cfg['db']['user'],
            password = cc.cfg['db']['pass'],
            host = cc.cfg['db']['host'],
            database= cc.cfg['db']['name'])
    # except:
        # print 'Unable to connect'
    return con


def getData():
    print('Downloading data...')

    con = getCon()

    query = 'SELECT {} FROM {}'.format(
        ','.join(['"{}"'.format(x) for x in cc.exp['fetch']['cols']]),
        cc.exp['fetch']['table'])

    if cc.exp['fetch']['where']:
        query += ' WHERE {}'.format(cc.exp['fetch']['where'])
    if cc.exp['fetch']['order']:
        query += ' ORDER BY {} '.format(cc.exp['fetch']['order'])
    if cc.exp['fetch']['limit']:
        query += ' LIMIT {}'.format(cc.exp['fetch']['limit'])

    print(query)

    df = pd.read_sql(
        sql = query,
        con = con,
        index_col = cc.exp['fetch']['index_col'])

    print('...done')
    return df

def sendStatistics(**kwargs):
    print('Sending statistics...')

    con = getCon()

    cols = []
    vals = []

    for col,val in kwargs.iteritems():
        cols.append(col)
        # mysql.connector has problems with converting numpy values, we supply explicit conversion

        if(type(val) is dict):
            if val['type']=='bin':
                if cc.cfg['db']['driver']=='postgresql':
                    vals.append(psycopg2.Binary(val['val']))
                else:
                    vals.append(val['val'])
            else:
                vals.append(val['val'])
        else:
            if type(val) is numpy.float64:
                vals.append(float(val))
            elif type(val) is numpy.int64:
                vals.append(int(val))
            else:
                vals.append(val)

    query = 'INSERT INTO {} ({}) VALUES ({})'.format(
        cc.cfg['statistics']['table'],
        ','.join(cols),
        ','.join(['%s']*len(cols)))

    print(query)

    cursor = con.cursor()

    try:
        cursor.execute(query, tuple(vals))
        con.commit()
    except Exception as e:
        con.rollback()
        print 'DB exception:', e

    cursor.close()
    con.close()

    print('...done')
