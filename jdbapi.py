import pandas as pd
import jaydebeapi as jdbc
import os

def read_jdbc(sql_statement, database = "TELCOANAPRD"):
    '''
    Description: query from Netezza
    Input: sql statement
    Implementation : read_jdbc('sql statement')
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
    # finally:
    #     curs.close()
    #     conn.close()


def query(sql,chunksize = None): # edit chunk size here
    '''
    Description: query from Netezza and put chunksize in if work as a iterator
    Input: sql statement and chunk size if any
    Implementation : query('sql statement', 500)
    '''
    arg = ['Kasidi3', '12345678']
    class_name = "org.netezza.Driver"
    conn_string = 'jdbc:netezza://10.50.78.21:5480/TELCOANAPRD'
    class_path = "/jdbc/nzjdbc.jar"
    
    try:
        conn = jdbc.connect(class_name, conn_string, arg, class_path, "")
        cursor = conn.cursor()
        return pd.read_sql(sql, conn,chunksize = chunksize)
    except jdbc.DatabaseError as de:
        raise
    # finally:
    #     cursor.close()
    #     conn.close()


def execute(sql):
    '''
    Description: execute
    Input: sql statement
    Implementation : 
    '''
    arg = ['Kasidi3', '12345678']
    class_name = "org.netezza.Driver"
    conn_string = 'jdbc:netezza://10.50.78.21:5480/TELCOANAPRD'
    class_path = "/jdbc/nzjdbc.jar"
    try:
        conn = jdbc.connect(class_name, conn_string, arg, class_path, "")
        cursor = conn.cursor()
        cursor.execute(sql)
    except jdbc.DatabaseError as de:
        raise
    # finally:
    #     cursor.close()
    #     conn.close()

def tocsv(pathname,sql):
    '''
    Description: create csv file
    Input: path/filename.csv, sql statement
    Implementation : 
    '''
    arg = ['Kasidi3', '12345678']
    class_name = "org.netezza.Driver"
    conn_string = 'jdbc:netezza://10.50.78.21:5480/TELCOANAPRD'
    class_path = "/jdbc/nzjdbc.jar"
    try:
        conn = jdbc.connect(class_name, conn_string, arg, class_path, "")
        cursor = conn.cursor()
        cursor.execute(f"""
                        CREATE EXTERNAL TABLE '{pathname}'
                        USING (delimiter '|' remotesource jdbc IncludeHeader True ENCODING 'internal') AS
                        {sql}
                            """)
    except jdbc.DatabaseError as de:
        raise
    # finally:
    #     cursor.close()
    #     conn.close()

# def fromcsv(pathname,tablename):
#     '''
#     Description: create csv file
#     Input: path/filename.csv, table name
#     Implementation : 
#     '''
#     arg = ['Kasidi3', '12345678']
#     class_name = "org.netezza.Driver"
#     conn_string = 'jdbc:netezza://10.50.78.21:5480/TELCOANAPRD'
#     class_path = "/jdbc/nzjdbc.jar"
#     try:
#         conn = jdbc.connect(class_name, conn_string, arg, class_path, "")
#         cursor = conn.cursor()
#         cursor.execute(f"""
#                         INSERT INTO {tablename}
#                         SELECT 
#                           *
#                         FROM EXTERNAL
#                           '{pathname}' 
#                         USING (delimiter '|' 
#                                 remotesource jdbc 
#                                 IncludeHeader True 
#                                 ENCODING 'internal')                        
#                             """)
#     except jdbc.DatabaseError as de:
#         raise
#     # finally:
#     #     cursor.close()
#     #     conn.close()

# def fromcsv(pathname,tablename):
#     '''
#     Description: create csv file
#     Input: path/filename.csv, table name
#     Implementation : 
#     '''
#     arg = ['Kasidi3', '12345678']
#     class_name = "org.netezza.Driver"
#     conn_string = 'jdbc:netezza://10.50.78.21:5480/TELCOANAPRD'
#     class_path = "/jdbc/nzjdbc.jar"
#     try:
#         conn = jdbc.connect(class_name, conn_string, arg, class_path, "")
#         cursor = conn.cursor()
#         cursor.execute(f"""
# DROP TABLE TMP_S IF EXISTS;
# CREATE EXTERNAL TABLE TMP_S
# (
# 	SUBS_KEY CHARACTER VARYING(20),
# 	IS_CHURN_7D INTEGER,
# 	IS_CHURN_30D INTEGER,
# 	PROB_CHURN_7D REAL,
# 	PROB_CHURN_30D REAL,
# 	CHURN_FACTOR_1_7D CHARACTER VARYING(50),
# 	CHURN_FACTOR_2_7D CHARACTER VARYING(50),
# 	CHURN_FACTOR_3_7D CHARACTER VARYING(50),
# 	CHURN_FACTOR_1_30D CHARACTER VARYING(50),
# 	CHURN_FACTOR_2_30D CHARACTER VARYING(50),
# 	CHURN_FACTOR_3_30D CHARACTER VARYING(50),
# 	CHURN_LIKELY7D CHARACTER VARYING(1),
# 	CHURN_LIKELY30D CHARACTER VARYING(1),
# 	KEY_DATE INTEGER
# )
# USING ( 
#   DATAOBJECT( '{pathname}' )  
#   REMOTESOURCE jdbc
#   DELIMITER '|');

# INSERT INTO {tablename}
# SELECT *
# FROM TMP_S
# DISTRIBUTE ON RANDOM
#                             """)
#     except jdbc.DatabaseError as de:
#         raise
#     # finally:
#     #     cursor.close()
#     #     conn.close()



def insertBulk(table, df): #insert from data frame pandas
    """
    Insert pandas df into Netezza
    insertBulk('tablename', dataframe object)
    """
    arg = ['Kasidi3', '12345678']
    class_name = "org.netezza.Driver"
    conn_string = 'jdbc:netezza://10.50.78.21:5480/TELCOANAPRD'
    class_path = "/jdbc/nzjdbc.jar"
    
       
    col_str = ",".join(list(df.iloc[0].index))
    val_str = ",".join(list(df.iloc[0].apply(lambda x: '?')))
    sql = """INSERT INTO {table} ({cols})
        VALUES ({vals}) """.format(table=table, cols=col_str, vals=val_str)
    values = []
    for index,row in df.iterrows():
        values.append(tuple(row))
    
    try:
        conn = jdbc.connect(class_name, conn_string, arg, class_path, "")
        cursor = conn.cursor()
        cursor.executemany(sql, values)
    except jdbc.DatabaseError as de:
        raise
    # finally:
    #     cursor.close()
    #     conn.close()
    