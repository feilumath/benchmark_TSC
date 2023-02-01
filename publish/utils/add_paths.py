import os


def add_paths(sub_name):
    """
    set data path to save data in a DIR different from code. Because code is connected to repo.
    Input:
        sub_name: the name of the sub-folder, eg. "moredim_case/".
    Output:
        Main_Data_path: homeDIR/project_name/data/
        Sub_Data_path: homeDIR/project_name/data/sub_name
        Sub_Results_path: homeDIR/project_name/results/sub_name
    """
    project_name = 'sigUCts'
    try:
        homeDIR      = os.environ['HOME']     # get homedir of your device
    except:
        homeDIR      = os.path.expanduser('~')             # get homedir of your device
    data_path    = homeDIR + '/'+project_name  # set data dir to save data, with project name
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    Main_Data_path    = data_path + '/data/'
    if not os.path.exists(Main_Data_path):
        os.makedirs(Main_Data_path)
    Main_Results_path = data_path + '/results/'
    if not os.path.exists(Main_Results_path):
        os.makedirs(Main_Results_path)
    Sub_Data_path     = Main_Data_path + sub_name
    if not os.path.exists(Sub_Data_path):
        os.makedirs(Sub_Data_path)
    Sub_Results_path  = Main_Results_path + sub_name
    if not os.path.exists(Sub_Results_path):
        os.makedirs(Sub_Results_path)
    return Main_Data_path, Sub_Data_path, Sub_Results_path
