"""
Eidolon 1.0
@since 2019.12.20
@version yuwei
该框架提供一个完整的训练环境，用于管理不同的训练过程。
"""


# system lib
import sys
import getopt
import os
import importlib

# inner lib
import config

help = """
Eidolon: A framework to manage your training.
@author yuwei
@since 2019.12.20

Usage: daemon <-options> [value] ...\n
including:\n
-e or --env       Current working environment, the default value is current dir.\n
-c or --config    The config file, the default value is config.json in current work directionary.\n
-s or --script    The running python script, the default value is train.py in root directionary.\n
-h or --help      Show the help document.
"""


def run(argv):
    """
    启动函数，旨在提供一个脚本执行入口。
    """
    # default runtime environment
    env = "./"

    # running script
    running_script = "../../train.py"

    # parse input arguments if exist
    try:
        options, _ = getopt.getopt(sys.argv[1:], "he:c:s:", [
            "help", "env=", "config=", "script="])
        print(options)

        for key, value in options:
            if key in ("-c", "--config"):
                config_file = value
            if key in ("-e", "--env"):
                env = value
            if key in ("-h", "--help"):
                print(help)
                sys.exit()
            if key in ("-s", "--script"):
                running_script = value

    except Exception:
        print("Error: No arguments found. Please see the document.")
        print(help)
        sys.exit()

    # set current runtime environment
    os.chdir(env)
    print("current working enivronment: {}".format(os.getcwd()))

    # check current python version
    print("current python version:{}".format(sys.version))

    # load config file
    config_loader = config.ConfigLoader(config_file)

    # 若输入的脚本带有.py后缀，自动去除
    if running_script.endswith(".py"):
        running_script = running_script.replace(".py", "")

    # load extern script
    module = importlib.import_module(running_script)

    #设置最大递归深度
    sys.setrecursionlimit(9000000)

    # try:
    # search main function
    main_func = getattr(module, "main")
    # start
    main_func(config_loader)
    # except AttributeError:
    #     print("Error: No main function found. Please define main(config) function in the script: {}.py".format(
    #         running_script))


if __name__ == "__main__":
    # start running
    run(sys.argv)
