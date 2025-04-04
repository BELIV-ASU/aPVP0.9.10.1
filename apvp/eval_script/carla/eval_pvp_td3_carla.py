#eval script for single carla ckpt change line 114 for pvp, old, TD3 model,
import argparse
import os
import os.path as osp
from collections import defaultdict

import pandas as pd
import json
import numpy as np
from pvp.sb3.common.monitor import Monitor
from pvp.eval_script.carla.carla_eval_utils import setup_model, setup_model_old
from pvp.experiments.carla.carla_env import HumanInTheLoopCARLAEnv


def load_human_data(path, data_usage=5000):
    """
   This method reads the states and actions recorded by human expert in the form of episode
   """
    with open(path, "r") as f:
        episode_data = json.load(f)["data"]
    np.random.shuffle(episode_data)
    assert data_usage < len(episode_data), "Data is not enough"
    data = {"state": [], "action": [], "next_state": [], "reward": [], "terminal": []}
    for cnt, step_data in enumerate(episode_data):
        if cnt >= data_usage:
            break
        data["state"].append(step_data["obs"])
        data["next_state"].append(step_data["new_obs"])
        data["action"].append(step_data["actions"])
        data["terminal"].append(step_data["dones"])
    # get images as features and actions as targets
    return data


def eval_one_checkpoint(model_path, model, eval_env, log_dir, num_episodes):
    model.set_parameters(model_path)
    count = 0
    step_data = defaultdict(list)
    episode_stats = [] 
    recorder = defaultdict(list)

    try:
        obs = eval_env.reset()
        episode_steering = []
        episode_throttle = []
        episode_speed = []
        episode_steering_au = []  # 记录辅助网络的 steering
        episode_throttle_au = []  # 记录辅助网络的 throttle

        while True:
            # 获取 action 和 action_au
            action, _states, action_au = model.predict(obs, deterministic=True)
            action_au = np.squeeze(action_au) 
            #print(f"[DEBUG] action_au: {action_au}, shape: {np.shape(action_au)}")
            obs, reward, done, info = eval_env.step(action)

            # 解析 action 和 action_au
            steering, throttle = action[0], action[1]
            steering_au, throttle_au = action_au[0], action_au[1]

            # ✅ 每一步都直接存入 step_data（避免使用 extend()）
            step_data["episode"].append(count)  
            step_data["steering"].append(steering)
            step_data["throttle"].append(throttle)
            step_data["speed"].append(info.get("speed", 0))
            step_data["steering_au"].append(steering_au)
            step_data["throttle_au"].append(throttle_au)


            # 记录 step 级别数据
            episode_steering.append(steering)
            episode_throttle.append(throttle) 
            episode_speed.append(info.get("speed", 0))
            episode_steering_au.append(steering_au)
            episode_throttle_au.append(throttle_au)

            if done:
                count += 1
                for k, v in info.items():
                    recorder[k].append(v)
                print("The environment is terminated. Final info: ", info)
                if count >= num_episodes:
                    break
                obs = eval_env.reset()

                # 计算该 episode 的均值
                mean_steering = np.mean(episode_steering)
                mean_throttle = np.mean(episode_throttle)
                mean_speed = np.mean(episode_speed)
                mean_steering_au = np.mean(episode_steering_au)  # 计算辅助网络 steering 均值
                mean_throttle_au = np.mean(episode_throttle_au)  # 计算辅助网络 throttle 均值

                # 记录该 episode 的均值
                episode_stats.append((mean_steering, mean_throttle, mean_speed, mean_steering_au, mean_throttle_au))

                episode_steering.clear()
                episode_throttle.clear()
                episode_speed.clear()
                episode_steering_au.clear()
                episode_throttle_au.clear()


        # # ===== 保存 episode 级别统计数据 =====
        # results = {
        #     "episode": list(range(1, num_episodes + 1)),
        #     "mean_steering": [s[0] for s in episode_stats],
        #     "mean_throttle": [s[1] for s in episode_stats],
        #     "mean_speed": [s[2] for s in episode_stats],
        #     "mean_steering_au": [s[3] for s in episode_stats],
        #     "mean_throttle_au": [s[4] for s in episode_stats],
        # }

        # print("[DEBUG] results:")
        # for key, value in results.items():
        #     print(f"{key}: length = {len(value)}")
            
        # df_episode = pd.DataFrame(results)
        # df_episode.to_csv(osp.join(log_dir, "eval_result.csv"), index=False)
        # print(f"✅ 评估结果已保存到 {osp.join(log_dir, 'eval_result.csv')}")
        # 清空当前 episode 的数据，开始记录下一个 episode

    finally:
        #print("[DEBUG] step_data content:", step_data)
        human_data=pd.DataFrame(eval_env.human_data)
        human_data.to_csv(osp.join(log_dir, "human_data.csv"), index=False)
        df_steps = pd.DataFrame(step_data)
        df_steps.to_csv(osp.join(log_dir, "step_data.csv"), index=False)
        print(f"✅ 逐步 action 和速度数据已保存到 {osp.join(log_dir, 'step_data.csv')}")
        eval_env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="./eval", type=str, help="CKPT name.")
    parser.add_argument("--port", default=9000, type=int, help="Carla server port.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    args = parser.parse_args()
    port = args.port
    seed = args.seed
    ckpt = args.ckpt

    num_episodes = 3
    obs_mode = "birdview"

    # ===== Setup the training environment =====
    train_env = HumanInTheLoopCARLAEnv(
        external_config=dict(
            obs_mode=obs_mode,
            force_fps=0,
            disable_vis=False,  # xxx: @xxx, change this to disable/open vis!
            debug_vis=False,
            port=port,
            disable_takeover=True,
            controller="keyboard",
            env={"visualize": {
                "location": "lower right"
            }}
        )
    )
    eval_env = Monitor(env=train_env, filename=None)
    model = setup_model(eval_env=eval_env, seed=seed, obs_mode=obs_mode)
    #setuo_model_old for old old, setup_modeltd3 for td3 baselines
    model_root_path = ckpt
    checkpoints = [p for p in os.listdir(model_root_path) if p.startswith("rl_model")]
    checkpoint_indices = sorted([int(p.split("_")[2]) for p in checkpoints], reverse=True)
    # eval_one_checkpoint(
    #     model=model, eval_env=eval_env, log_dir="../", num_episodes=num_episodes
    # )
    # for model_index in checkpoint_indices[::2]:
    for model_index in [60000]:
        model_path = os.path.join(model_root_path, "rl_model_{}_steps.zip".format(model_index))
        log_dir = model_path.replace(".zip", "")
        os.makedirs(log_dir, exist_ok=True)
        eval_one_checkpoint(
            model_path=model_path, model=model, eval_env=eval_env, log_dir=log_dir, num_episodes=num_episodes
        )
