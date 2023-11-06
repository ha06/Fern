import os
import shutil

# Here, we get the current working directory: This will enable us navigate to output & input folders as well as another PC
curr_wrk_dir = os.getcwd()
# print(curr_wrk_dir)
# Here, we assigin a variable to the output directory
root_output_dir = curr_wrk_dir + '/output/'


# print(root_output_dir)

def clear_directory(directory_to_clear):
    result = False

        # list all directories current available in output to delete
    print("directory_to_clear", directory_to_clear)
    if os.path.exists(directory_to_clear) is True:
        y = input("press y/Y to clear directory: ")
        if y == 'y' or y == 'Y':
            shutil.rmtree(directory_to_clear)
            os.mkdir(directory_to_clear)
            print('Clear output in directory {0}'.format(directory_to_clear))
            return True
        else:
            return False
    os.mkdir(directory_to_clear)
    return True


def log_initial_pop(output_dir, pop, cost, itr_cnt):
    # create directory with iteration number
    # os.mkdir(root_output_dir + str(it))
    # output_dir =  root_output_dir + str(it) +'/'
    # print(output_dir)
    # print(str(it))
    output_path = os.path.join(output_dir, 'initial_population_' + itr_cnt + '.csv')
    try:
        if os.path.isfile(output_path):
            f = open(output_path, 'a')
            text = pop + ' Cost:' + cost + '\n'
            f.write(text)
            f.close()
        else:
            f = open(output_path, 'w+')
            text = 'Initial population set for iteration ' + itr_cnt + ': \n'
            f.write(text)
            text = pop + ' Cost:' + cost + '\n'
            f.write(text)
            f.close()
    except Exception as e:
        print("Error logging path to file")
        print(e)


def log_sorted_pop(output_dir, pop, cost, itr_cnt):
    output_path = os.path.join(output_dir, 'sorted_before_crossover_' + itr_cnt + '.csv')
    try:
        if os.path.isfile(output_path):
            f = open(output_path, 'a')
            text = pop + ' Cost:' + cost + '\n'
            f.write(text)
            f.close()
        else:
            f = open(output_path, 'w+')
            text = 'Sorted population set for iteration ' + itr_cnt + ': \n'
            f.write(text)
            text = pop + ' Cost:' + cost + '\n'
            f.write(text)
            f.close()
    except Exception as e:
        print("Error logging path to file")
        print(e)


def log_crossover_pop(output_dir, pop, cost, itr_cnt):
    output_path = os.path.join(output_dir, 'crossedover_population_' + itr_cnt + '.csv')
    try:
        if os.path.isfile(output_path):
            f = open(output_path, 'a')
            text = pop + ' Cost:' + cost + '\n'
            f.write(text)
            f.close()
        else:
            f = open(output_path, 'w+')
            text = 'Population after crossover: ' + '\n'
            f.write(text)
            text = pop + ' Cost:' + cost + '\n'
            f.write(text)
            f.close()
    except Exception as e:
        print("Error logging path to file")
        print(e)


def log_mutation_pop(output_dir, pop, cost, itr_cnt):
    output_path = os.path.join(output_dir, 'mutated_population_' + itr_cnt + '.csv')
    try:
        if os.path.isfile(output_path):
            f = open(output_path, 'a')
            text = pop + ' Cost:' + cost + '\n'
            f.write(text)
            f.close()
        else:
            f = open(output_path, 'w+')
            text = 'Population after mutation: ' + '\n'
            f.write(text)
            text = pop + ' Cost:' + cost + '\n'
            f.write(text)
            f.close()
    except Exception as e:
        print("Error logging path to file")
        print(e)


def log_final_pop(output_dir, pop, cost, itr_cnt):
    output_path = os.path.join(output_dir, 'final_population_' + itr_cnt + '.csv')
    try:
        if os.path.isfile(output_path):
            f = open(output_path, 'a')
            text = pop + ' Cost:' + cost + '\n'
            f.write(text)
            f.close()
        else:
            f = open(output_path, 'w+')
            text = 'Final set of population: ' + '\n'
            f.write(text)
            text = pop + ' Cost:' + cost + '\n'
            f.write(text)
            f.close()
    except Exception as e:
        print("Error logging path to file")
        print(e)

# if __name__ == '__main__':
# 	x1 = [0.21651, 0.35631, 0.36532, 0.35261, 0.40381, 0.36531, 0.30663, 0.27358]
# 	x2 = [0.50999, 0.39681, 0.30929, 0.37265, 0.26156, 0.34083, 0.29692, 0.312  ]
# 	log_initial_pop (str(x1), 1 )
