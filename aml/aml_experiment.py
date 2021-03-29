#import mlflow
import os
import argparse
from utils import get_config
from azureml.core import Experiment
from azureml.core import Workspace
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core.compute import ComputeInstance, ComputeTarget
from azureml.core.runconfig import PyTorchConfiguration
from azureml.core.runconfig import MpiConfiguration


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file',default='config.yaml',type=str)

    args = parser.parse_args()
    cfg = get_config(args.config_file)

    experiment_name = cfg['aml']['experiment_name']
    #access aml workspace
    ws = Workspace.from_config(cfg['aml']['aml_config'])

    #create experiment
    experiment = Experiment(workspace=ws,
            name=experiment_name)

    #create environment
    myenv = Environment.from_pip_requirements(name='b2b_torch',
            file_path='requirements.txt')

    #access existing compute target
    if cfg['aml']['compute']['use_instance']:
        compute_target = ComputeInstance(
                workspace=ws,
                name=cfg['aml']['compute']['name']
                )
    else:
        compute_target = ComputeTarget(
                workspace=ws,
                name=cfg['aml']['compute']['name']
                )

    #access existing dataset
    dataset = Dataset.get_by_name(ws, name=cfg['aml']['dataset'])

    #setup up for distributed training
    distr_config = PyTorchConfiguration(
                    process_count=cfg['aml']['process_count'],
                    node_count=cfg['aml']['node_count']
                    )

    #create training run
    src = ScriptRunConfig(source_directory='./',
                        script='train_net.py',
                        compute_target=compute_target,
                        distributed_job_config=distr_config,
                        environment=myenv,
                        arguments=['--data-path',dataset.as_named_input('input').as_mount(),
                                    '--local_rank','$LOCAL_RANK'])

    run = experiment.submit(config=src)
    run.wait_for_completion(show_output=True)
