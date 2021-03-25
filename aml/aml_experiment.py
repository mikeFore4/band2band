import os
from azureml.core import Experiment
from azureml.core import Workspace
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core.compute import ComputeInstance, ComputeTarget
from azureml.core.runconfig import PyTorchConfiguration
from azureml.core.runconfig import MpiConfiguration

#access aml workspace
ws = Workspace.from_config('aml/aml_config.json')

#create experiment
experiment_name = 'test_run'
experiment = Experiment(workspace=ws,
        name=experiment_name)

#create environment
myenv = Environment.from_pip_requirements(name='b2b_torch',
        file_path='requirements.txt')
#curated_env_name = 'AzureML-PyTorch-1.6-GPU'
#myenv = Environment.get(workspace=ws, name=curated_env_name)

#access existing compute target
#compute_target = ComputeInstance(workspace=ws, name='mf-synth-gpu-cluster')
compute_target = ComputeTarget(workspace=ws, name='gpu-cluster')

#access existing dataset
dataset = Dataset.get_by_name(ws, name='b2b')
#ws.set_default_datastore('band2band')
#datastore = ws.get_default_datastore()
#dataset = Dataset.File.from_files(path=(datastore, 'datasets/b2b'))

#setup up for distributed training
#distr_config = PyTorchConfiguration(node_count=1)
#launch_cmd = f"python -m torch.distributed.launch --nproc_per_node=4 train_net.py --data-path={dataset.as_mount()}".split()
#distr_config = MpiConfiguration(process_count_per_node=4, node_count=1)
distr_config = PyTorchConfiguration(process_count=4, node_count=2)

#create training run
src = ScriptRunConfig(source_directory='/home/mifore/work/synth/adagan',
                    script='train_net.py',
                    compute_target=compute_target,
                    distributed_job_config=distr_config,
                    environment=myenv,
                    arguments=['--data-path',dataset.as_named_input('input').as_mount(),
                                '--local_rank','$LOCAL_RANK'])

run = experiment.submit(config=src)
run.wait_for_completion(show_output=True)



