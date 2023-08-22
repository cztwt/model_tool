import pandas as pd
import os
import logging
import warnings

from abc import ABC, abstractmethod
from pyhive import hive
import pymysql

warnings.filterwarnings('ignore')
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseLoader(ABC):
    def __init__(self, host, port, database, username, password):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.connection = None
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def disconnect(self):
        pass
    
    @abstractmethod
    def load_data(self, table_name):
        pass

class HiveDataLoader(DatabaseLoader):
    def connect(self):
        try:
            self.connection = hive.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                username=self.username,
                password=self.password
            )
            logger.info('成功连接hive数据库!')
        except Exception as e:
            logger.info('连接hive数据库错误: {}'.format(e))
    
    def disconnect(self):
        if self.connection:
            self.connection.close()
            logger.info('已断开hive数据库连接信息。')
    
    def load_data(self, query):
        if not self.connection:
            logger.info('请先连接hive数据库.')
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(data=data, columns=columns)
        except Exception as e:
            logger.info('查询错误：{}'.format(e))

class MySQLDataLoader(DatabaseLoader):
    def connect(self):
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database
            )
            logger.info('成功连接mysql数据库!')
        except Exception as e:
            logger.info('连接mysql数据库错误: {}'.format(e))
    
    def disconnect(self):
        if self.connection:
            self.connection.close()
            logger.info('已断开mysql数据库连接信息。')
    
    def load_data(self, query):
        if not self.connection:
            logger.info('请先连接mysql数据库.')
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            data = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return pd.DataFrame(data=data, columns=columns)
        except Exception as e:
            logger.info('查询错误：{}'.format(e))
