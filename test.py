import logging
import sys

# logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(funcName)s - %(module)s - %(message)s")

# def hello():
#     logging.info("hello")
#     logging.error("error")

def hello():
    print(sys._getframe().f_code.co_name)
    print(sys._getframe().f_code.co_filename)

