import argparse
import os
import os.path as osp
from collections import defaultdict
import numpy as np
import pandas as pd
from pvp.sb3.common.monitor import Monitor
from pvp.eval_script.carla.carla_eval_utils import setup_model
from pvp.experiments.carla.carla_env import HumanInTheLoopCARLAEnv


def eval_checkpoint_single_episode(model, eval_env, num_episodes=2):
    """
    评估模型，并记录 steering、throttle、速度，确保 reset 之后的初始化位置一致。
    """
    all_data = {"steering": [], "throttle": [], "speed": []}  # 记录所有 episode 的数据
    episode_stats = []  # 记录每个 episode 的均值

    for ep in range(num_episodes):
        obs = eval_env.reset()  # 确保 reset 后回到相同位置
        episode_steering = []
        episode_throttle = []
        episode_speed = []

        while True:
            action, _states = model.predict(obs, deterministic=True)  # 预测动作
            obs, reward, done, info = eval_env.step(action)

            # 记录 steering、throttle 和 速度
            episode_steering.append(action[0])  # 方向盘
            episode_throttle.append(action[1])  # 油门/加速
            episode_speed.append(info.get("speed", 0))  # 获取速度，默认 0

            if done:
                print(f"✅ Episode {ep+1} 结束")
                break

        # 记录当前 episode 的数据
        all_data["steering"].append(episode_steering)
        all_data["throttle"].append(episode_throttle)
        all_data["speed"].append(episode_speed)

        # 计算当前 episode 的平均 steering、throttle、速度
        mean_steering = np.mean(episode_steering)
        mean_throttle = np.mean(episode_throttle)
        mean_speed = np.mean(episode_speed)
        episode_stats.append((mean_steering, mean_throttle, mean_speed))

    return all_data, episode_stats


def compute_differences(stats):
    """
    计算两个 episode 之间的均值差异。
    """
    ep1, ep2 = stats  # 获取两个 episode 的均值数据
    steering_diff = abs(ep1[0] - ep2[0])
    throttle_diff = abs(ep1[1] - ep2[1])
    speed_diff = abs(ep1[2] - ep2[2])

    return {
        "steering_diff": steering_diff,
        "throttle_diff": throttle_diff,
        "speed_diff": speed_diff,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str, help="CKPT 模型路径，例如 /home/xxx/model/rl_model_1000_steps.zip")
    parser.add_argument("--port", default=9000, type=int, help="Carla server port.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    args = parser.parse_args()

    port = args.port
    seed = args.seed
    ckpt_path = args.ckpt

    num_episodes = 2  # 只跑 2 个 episode
    obs_mode = "birdview"

    # ===== 设置环境 =====
    train_env = HumanInTheLoopCARLAEnv(
        config=dict(
            obs_mode=obs_mode,
            force_fps=0,
            disable_vis=False,
            debug_vis=False,
            port=port,
            disable_takeover=True,
            controller="keyboard",
            env={"visualize": {"location": "lower right"}}
        )
    )
    eval_env = Monitor(env=train_env, filename=None)

    # ===== 加载模型 =====
    model = setup_model(eval_env=eval_env, seed=seed, obs_mode=obs_mode)
    model.set_parameters(ckpt_path)

    # ===== 评估模型 =====
    all_data, episode_stats = eval_checkpoint_single_episode(model, eval_env, num_episodes)

    # ===== 计算两个 episode 之间的差异 =====
    differences = compute_differences(episode_stats)
    print("📊 两个 episode 之间的均值差异:", differences)

    # ===== 存储结果到 CSV =====
    results = {
        "episode_1_mean_steering": [episode_stats[0][0]],
        "episode_2_mean_steering": [episode_stats[1][0]],
        "episode_1_mean_throttle": [episode_stats[0][1]],
        "episode_2_mean_throttle": [episode_stats[1][1]],
        "episode_1_mean_speed": [episode_stats[0][2]],
        "episode_2_mean_speed": [episode_stats[1][2]],
        "steering_diff": [differences["steering_diff"]],
        "throttle_diff": [differences["throttle_diff"]],
        "speed_diff": [differences["speed_diff"]],
    }

    df = pd.DataFrame(results)
    log_dir = ckpt_path.replace(".zip", "_eval")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = osp.join(log_dir, "eval_result.csv")
    df.to_csv(csv_path, index=False)

    print(f"✅ 评估结果已保存到 {csv_path}")
