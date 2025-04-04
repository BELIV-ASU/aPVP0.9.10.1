"""
Training script for training PVP in CARLA environment.
"""
import argparse
import os
from pathlib import Path

from pvp.experiments.carla.carla_env import HumanInTheLoopCARLAEnv
from pvp.pvp_td3 import PVPTD3
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.sac.our_features_extractor import OurFeaturesExtractor
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str
import wandb

# def main():

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="epvp_carla", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    parser.add_argument("--wandb_run_id", type=str, default="", help="The run ID for resuming in wandb.")
    parser.add_argument("--pretrained_model", type=str, default="", help="Path to the pretrained model.")


    parser.add_argument(
        "--obs_mode",
        default="birdview",
        choices=["birdview", "first", "birdview42", "firststack"],
        help="The observation mode."
    )
    parser.add_argument("--port", default=9000, type=int, help="Carla server port.")
    args = parser.parse_args()

    # ===== Set up some arguments =====
    port = args.port
    obs_mode = args.obs_mode
    if obs_mode.endswith("stack"):
        other_feat_dim = 0
    else:
        other_feat_dim = 1

    experiment_batch_name = "{}_{}".format(args.exp_name, obs_mode)
    seed = args.seed
    trial_name = "{}_{}".format(experiment_batch_name, get_time_str())

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    run_id = args.wandb_run_id
    
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    experiment_dir = Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(
            obs_mode=obs_mode,
            force_fps=10,
            disable_vis=False,
            port=port,
            enable_takeover=True,
            env=dict(visualize=dict(location="center"))
        ),

        # Algorithm config
        algo=dict(
            use_balance_sample=True,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,  # We run in reward-free manner!
            ),

            # PZH Note: Compared to MetaDrive, we use CNN as the feature extractor.
            # policy_kwargs=dict(net_arch=[256, 256]),
            policy_kwargs=dict(
                features_extractor_class=OurFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=256 + other_feat_dim),
                share_features_extractor=True,  # PZH: Using independent CNNs for actor and critics
                net_arch=[
                    256,
                ]
            ),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=100_000,  # We only conduct experiment less than K steps
            learning_starts=100,  # The number of steps before
            batch_size=128,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )

    # ===== Setup the training environment =====
    train_env = HumanInTheLoopCARLAEnv(external_config=config["env_config"], )
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Setup the callbacks =====
    save_freq = 500  # Number of steps per model checkpoint
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=1, save_freq=save_freq, save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    model = PVPTD3(**config["algo"])

    # ===== Load pretrained model if specified =====
    if args.pretrained_model:
        if os.path.exists(args.pretrained_model):
            model.load(args.pretrained_model)
            print(f"Loaded pretrained model from {args.pretrained_model}")
        else:
            print(f"Pretrained model path {args.pretrained_model} does not exist!")


    # Initialize Wandb
    if use_wandb:
        if run_id:
            wandb.init(
                project=project_name,
                entity=team_name,
                config=config,
                resume="allow",
                id=run_id,
            )
        else:
            wandb.init(
                project=project_name,
                entity=team_name,
                config=config,
            )
    
    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=100_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=None,
        eval_freq=-1,
        n_eval_episodes=2,
        eval_log_path=None,

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )
