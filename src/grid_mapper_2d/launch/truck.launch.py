import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('grid_mapper_2d')
    default_cfg  = os.path.join(pkg_share, 'config', 'truck.yaml')
    default_rviz = os.path.join(pkg_share, 'rviz',   'grid_mapper_2d.rviz')

    config_file = LaunchConfiguration('config_file')
    rviz_cfg    = LaunchConfiguration('rviz_cfg')
    use_rviz    = LaunchConfiguration('rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_cfg = DeclareLaunchArgument(
        'config_file',
        default_value=default_cfg,
        description='Path to grid_mapper_2d yaml.'
    )
    declare_rviz_cfg = DeclareLaunchArgument(
        'rviz_cfg',
        default_value=default_rviz,
        description='RViz layout file.'
    )
    declare_use_rviz = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to launch rviz2.'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use /clock from ros2 bag play.'
    )

    mapper_node = Node(
        package='grid_mapper_2d',
        executable='grid_mapper_2d_node',
        name='grid_mapper_2d',
        output='screen',
        parameters=[config_file, {'use_sim_time': use_sim_time}],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
        output='log',
    )

    ld = LaunchDescription()
    ld.add_action(declare_cfg)
    ld.add_action(declare_rviz_cfg)
    ld.add_action(declare_use_rviz)
    ld.add_action(declare_use_sim_time)
    ld.add_action(mapper_node)
    ld.add_action(rviz_node)
    return ld


