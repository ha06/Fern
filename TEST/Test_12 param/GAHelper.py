from __future__ import division

import os
import  shutil

def clear_directory(directory_to_clear):

    result  = True

    try:
        # list all directories current available in output to delete
        file_list = os.listdir(directory_to_clear)
        #arr_txt = [x for x in os.listdir() if x.endswith(".txt")]
        # delete all files present
        # if len(file_list) > 0:
        #
        #     for folder in file_list:
        #         if 'Store' not in str(folder):
        #             shutil.rmtree(directory_to_clear + folder)
        #         #arr_txt = [x for x in os.listdir() if x.endswith(".txt")]
        #     # for file in file_list:
        #         #if len(arr_txt) > 0:
        #             #shutil.rmtree(directory_to_clear + str(arr_txt))
        print("directory_to_clear", directory_to_clear)
        if len(file_list) > 0:

            for folder in file_list:
                if 'Store' not in str(folder):
                    shutil.rmtree(directory_to_clear + folder)

            print('Clear output in directory {0}'.format(directory_to_clear))



    except Exception as e:
        print(e)
        print('Error while clearing directory')
        result = False

    return result
