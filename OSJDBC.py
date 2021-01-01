import pandas as pd
import jaydebeapi as jdbc
import os

def read_jdbc(sql_statement, database = "TELCOANAPRD"):
    '''
    read jdbc into panda dataframe
    '''
    #init
    dsn_database = database
    dsn_hostname = "10.50.78.21" 
    dsn_port = "5480"                
    dsn_uid = "Kasidi3"        
    dsn_pwd = "12345678"      
    jdbc_driver_name = "org.netezza.Driver"
    jdbc_driver_loc = os.path.join('C:\\JDBC\\nzjdbc.jar')
    connection_string='jdbc:netezza://'+dsn_hostname+':'+dsn_port+'/'+dsn_database
    try:
        conn = jdbc.connect(jdbc_driver_name, connection_string, {'user': dsn_uid, 'password': dsn_pwd},jars=jdbc_driver_loc)
    except jdbc.DatabaseError as de:
        raise 
    try:
        curs = conn.cursor()
        curs.execute(sql_statement)
        columns = [desc[0] for desc in curs.description] #getting column headers
        #convert the list of tuples from fetchall() to a df
        return pd.DataFrame(curs.fetchall(), columns=columns) 
    except jdbc.DatabaseError as de:
        raise
    finally:
        curs.close()
        conn.close()

os.environ['JAVA_HOME']='C:\\Program Files\\Java\\jdk1.8.0_251\\bin'#path java jdk
conn = jdbc.connect('org.netezza.Driver',
    'jdbc:netezza://10.50.78.21:5480/TELCOANAPRD',
    ['Kasidi3', '12345678'],
    'C:\\JDBC\\nzjdbc.jar',)#path file jdbc 
cursor = conn.cursor()

def query(sql):
    return pd.read_sql(sql, conn)
def execute(sql):
    cursor.execute(sql)
def insertBulk(table, df): #insert from data frame pandas
    """
    Insert pandas df into Netezza
    insertBulk('tablename', dataframe object)
    """
    col_str = ",".join(list(df.iloc[0].index))
    val_str = ",".join(list(df.iloc[0].apply(lambda x: '?')))
    sql = """INSERT INTO {table} ({cols})
        VALUES ({vals}) """.format(table=table, cols=col_str, vals=val_str)
    values = []
    for index,row in df.iterrows():
        values.append(tuple(row))
    cursor.executemany(sql, values)