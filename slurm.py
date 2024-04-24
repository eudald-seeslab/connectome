#!/bin/python3

import subprocess

BASE_PATH = '/home/eudald/connectome'
NODES_PER_TASK = 1
PROC_PER_TASK = 1
USER_MAIL = 'alejandro.horrillo@urv.cat'
JOB_NAME = 'Eudald_jobs'
OUTPUT_PATH = BASE_PATH + '/logs_python_cluster.txt'
COMMAND_PATH = BASE_PATH + '/eudald_venv/bin/python3'
SCRIPT_PATH = BASE_PATH + '/full_training_layers.py'

# Genera una lista de strings que contiene los argumentos para el proceso.
def generate_arguments():
        args=[[4,x] for x in range(20)] + [[5,x] for x in range(20)]
        #args=[[4 ,0], [4, 1]]
#	args = np.random.rand(10000, 1)
        return args

# Construye el srun.
def build_command(arg):
	base_command = f'srun -N{NODES_PER_TASK} --ntasks-per-node {PROC_PER_TASK} --mail-user {USER_MAIL} -J {JOB_NAME} --mail-type=ALL --error={OUTPUT_PATH} --output={OUTPUT_PATH} '
	script_command = f'{COMMAND_PATH} {SCRIPT_PATH} {arg[0]} {arg[1]}'
	return base_command + script_command

def main():
	args = generate_arguments()
	if len(args) == 0:
		print("no arguments passed")
		command = build_command('')
		process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
		#output, error = process.communicate()
	else:
		for arg in args:
			command = build_command(arg)
			process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
			#output, error = process.communicate()

if __name__ == "__main__":
	main()
