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
from azureml.train.hyperdrive import HyperDriveConfig
from azureml.train.hyperdrive import uniform, choice
from azureml.train.hyperdrive import RandomParameterSampling
from azureml.train.hyperdrive import TruncationSelectionPolicy
from azureml.train.hyperdrive import PrimaryMetricGoal



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
                        arguments=['--data_path',dataset.as_named_input('input').as_mount(),
                                    '--local_rank','$LOCAL_RANK'])

    # set up hyperparameter tuning
    hp_space = {
            'batch_size': choice(16,32,48),
            'matching_weight':uniform(0.0,1.0),
            'rec_gen_weight':uniform(0.0,1.0),
            'rec_self_weight':uniform(0.0,1.0),
            'const_blocks': choice(range(1,4)),
            'up_down_blocks':choice(range(1,4)),
            'pooling_factor':choice(range(1,4)),
            'up_down_multiplier':choice(range(1,4)),
            'const_mult':choice(range(1,4)),
            'optimizer':choice('adam','sgd'),
            'learning_rate':uniform(.00001,.2),
            'momentum': uniform(0.0,2.0)
            }

    param_sampling = RandomParameterSampling(hp_space)

    early_termination_policy = TruncationSelectionPolicy(
                    truncation_percentage=20,
                    evaluation_interval=10,
                    delay_evaluation=5)

    hd_config = HyperDriveConfig(run_config=src,
                                hyperparameter_sampling=param_sampling,
                                policy=early_termination_policy,
                                primary_metric_name='Validation_loss/generative_reconstruction',
                                primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                                max_total_runs=30,
                                max_concurrent_runs=3)

    run = experiment.submit(hd_config)
    run.wait_for_completion(show_output=True)
