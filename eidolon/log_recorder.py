"""
初版日志记录器，还很不完善。目前调用logging模块。
之所以要封装这层，是为了避免代码入侵太强。
@since 2019.12.27
@author yuwei
"""

import logging


def init_log():
    """
    初始化log记录器
    """
    #创建基本的日志配置
    logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

def info(msg):
    """
    打印基本信息
    """
    


def error(msg):
    """
    错误信息
    """

