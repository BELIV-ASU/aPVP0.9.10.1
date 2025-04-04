import carla
import matplotlib.pyplot as plt

def draw_map_with_locations(client, town="Town01", locations=((0, 0), (1, 1))):
    """
    画出 CARLA 中的地图，并在上面标记指定的 waypoint 位置（通过坐标）。
    修正镜像问题，使 Y 轴方向与 CARLA 一致。

    :param client: CARLA 客户端
    :param town: 需要加载的地图
    :param locations: 需要标记的 waypoint 位置 ((start_x, start_y), (end_x, end_y))
    """
    # 连接 CARLA 并获取世界信息
    world = client.get_world()
    carla_map = world.get_map()

    # 获取地图上的所有 waypoints
    waypoints = carla_map.generate_waypoints(2.0)  # 2.0 表示每隔 2 米生成一个 waypoint

    # 提取所有 waypoints 的位置
    x_coords = [wp.transform.location.x for wp in waypoints]
    y_coords = [-wp.transform.location.y for wp in waypoints]  # 取反，修正镜像问题

    # 获取指定位置
    start_x, start_y = locations[0]
    end_x, end_y = locations[1]

    # 画图
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, s=3, color='gray', label="Waypoints")
    plt.plot([], [], color='green', label="Route")
    plt.scatter(start_x, -start_y, s=300, color='red', marker="*", label="Start Location")  # 取反
    plt.scatter(end_x, -end_y, s=300, color='blue', marker="o", label="End Location")  # 取反

    #plt.xlabel("X Coordinate (meters)")
    #plt.ylabel("Y Coordinate (meters) (Reversed for CARLA)")
    #plt.title(f"CARLA Map: {town} with Locations (Fixed)")
    #plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 连接 CARLA 服务器（请确保 CARLA Simulator 正在运行）
    client = carla.Client("localhost", 9000)
    client.set_timeout(10.0)  # 设置超时时间

    # 选择地图和 waypoint 位置
    town_name = "Town01"  # 你可以换成 "Town02", "Town03" 等
    start_location = (92.110001, 81.831619)  # 你可以替换为你的实际坐标
    end_location = (268.586578, 2.020086)

    # 画出地图和 waypoint
    draw_map_with_locations(client, town=town_name, locations=(start_location, end_location))
