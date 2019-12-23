#system lib
import os
import sys
import getopt
import shutil
import platform

#inner lib
from eidolon import config

def create_train_bootstrap(running_script):
    """
    create train bootstrap
    :param running_script
    :return:
    """

    # 运行指令
    run_cmd = "python ../../main.py --env=./ --config=config.json --script={}".format(running_script)

    # 检查当前系统，如果是winsows系统，则生成bat; 如果是linux系统，则生成sh
    system = platform.system()

    if system == "Windows":
        # windows中使用\r\n换行
        cmd = [run_cmd+"\r\n", "@pause\r\n"]
        with open("train.bat", "w") as f:
            f.writelines(cmd)

    elif system == "Linux":
        # linux中使用\n换行
        cmd = [run_cmd+"\n"]
        with open("train.sh", "w") as f:
            f.writelines(cmd)
    else:
        print("Error: System: {} is not supported currently.".format(system))
        sys.exit()


def create_config_bootstrap(app_name):
    """
    用于生成配置界面启动脚本, 配置界面功能已启用，不需要盯着晦涩的配置文件修改了。
    @since 2019.12.3
    @author yuwei
    """

    # 此命令用于隐藏cmd的黑框
    invisible_cmd = ['@echo off\r\n',
                     'if "%1" == "h" goto begin\r\n',
                     'mshta vbscript:createobject("wscript.shell").run("%~nx0 h",0)(window.close)&&exit\r\n',
                     ':begin\r\n']

    # 运行cmd
    run_cmd = "python ../../gui/ConfigUI/bootstrap.py -a {}\r\n".format(
        app_name)

    with open("config.bat", "w") as f:
        f.writelines(invisible_cmd)
        f.writelines([run_cmd])


def create_app(app_name, running_script, conf):
    """
    create an empty app
    :param app_name:
    :param running_script
    :param conf
    :return:
    """
    # create dictionary
    app_dir = os.path.join("./app/", app_name)

    # check is existed
    if os.path.exists(app_dir):
        return "error app name, the app = {} is already existed".format(app_name)

    os.mkdir(app_dir)

    # switch to app dir
    os.chdir(app_dir)

    # config_file="config.json"
    # create config file temple
    config.create_config_JSON_temple(conf)


    # create bootstrap train.bat
    create_train_bootstrap(running_script)
    # create config.bat
    create_config_bootstrap(app_name)
    # create paint script paint_loss.bat
    with open("paint_loss.bat", "w") as f:
        f.writelines(["python ../../paint_loss.py"])

    return "create app successfully, name={}, script={}, confing={}".format(app_name, running_script, conf)



help="""
Create Training Application.

@author yuwei
@since 2019.1.18

Usage: create <-options> [value] ...\n
including:\n
-n or --name      App name. Necessary.\n
-c or --config    The config file, the default value is config.json in current work directionary.\n
-s or --script    The running python script, the default value is train.py in root directionary.\n
-h or --help      Show the help document.
"""



if __name__ == "__main__":
    # parse input arguments if exist
    # 应用名
    app_name = None
    # 运行脚本
    running_script = "eidolon.train.py"
    # 默认推荐使用JSON格式
    conf = "config.json"

    try:
        options, _ = getopt.getopt(sys.argv[1:], "hn:w:d:s:c:", [
                                   "help", "name=", "script=", "config="])
        print(options)
        for key, value in options:
            if key in ("-n", "--name"):
                app_name = value
            if key in ("-s", "--script"):
                running_script = value
            if key in ("-c", "--config"):
                conf = value

    except Exception:
        print("Error: No arguments found. Please see the document.")
        print(help)
        sys.exit()  

    if app_name is not None:
        msg = create_app(app_name, running_script, conf)
        print(msg)
    else:
        print("no name is selected, please enter you name by --name= or -n ")
