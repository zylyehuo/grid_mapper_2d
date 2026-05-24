# 2D 栅格地图构建

## 效果图

![image-20260524100037362](/home/zylyehuo/mapping/grid_mapper_2d/assets/image-20260524100037362.png)

## 基本配置

> 需要话题：雷达话题、IMU话题

> cloud_topic: ""  # 雷达点云话题
> imu_topic:   ""  # IMU话题
> save_dir:    ""   # 地图保存路径

## 运行指令

```bash
cd ~/grid_mapper_2d
source ./install/setup.bash
ros2 launch grid_mapper_2d grid_mapper_2d.launch.py
```

```bash
ros2 bag play 录制的BAG/实时数据
```

