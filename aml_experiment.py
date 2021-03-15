import os
from azureml.core import Experiment
from azureml.core import Workspace
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core.compute import ComputeInstance
from azureml.core.runconfig import PyTorchConfiguration

#access aml workspace
ws = Workspace.from_config('aml_config.json')

#create experiment
experiment_name = 'test_run'
experiment = Experiment(workspace=ws,
        name=experiment_name)

#create environment
myenv = Environment.from_pip_requirements(name='b2b_torch',
        file_path='requirements.txt')

#access existing compute target
instance = ComputeInstance(workspace=ws, name='mf-synth-gpu-cluster')

#access existing dataset
dataset = Dataset.get_by_name(ws, name='b2b')
#ws.set_default_datastore('band2band')
#datastore = ws.get_default_datastore()
#dataset = Dataset.File.from_files(path=(datastore, 'datasets/b2b'))

#setup up for distributed training
distr_config = PyTorchConfiguration(node_count=1)
#launch_cmd = f"python -m torch.distributed.launch --nproc_per_node=4 train_net.py --data-path={dataset.as_mount()}".split()

#create training run
src = ScriptRunConfig(source_directory='/home/mifore/work/synth/adagan',
                    script='train_net.py',
                    compute_target=instance,
                    distributed_job_config=distr_config,
                    environment=myenv,
                    arguments=['--data-path',dataset.as_named_input('input').as_mount()])

run = experiment.submit(config=src)
run.wait_for_completion(show_output=True)



