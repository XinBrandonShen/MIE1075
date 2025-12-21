#!/usr/bin/env python3
"""
ROS 2 launch file for MTC pick and place with GraspNet integration (Multi-Grasp).

This launch file starts the MTC node with:
- Multiple GraspNet grasp poses (top 5)
- Alternatives container to try all poses
- RViz with Motion Planning Tasks panel
- Fallback to center-based grasping

Launch Arguments:
    robot_name: Name of the robot (default: mycobot_280)
    use_sim_time: Use simulation clock (default: true)
    execute: Execute planned trajectory (default: false)
    use_graspnet: Use GraspNet for grasps (default: true)
    fallback_to_center: Fallback to center if GraspNet fails (default: true)
    num_grasp_candidates: Number of grasp poses to try (default: 5)
    place_x, place_y, place_z: Place position coordinates
    launch_rviz: Launch RViz with MTC panel (default: false)

:author: Brandon & Claude
:date: December 2024
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """Generate launch description."""
    
    # Package names
    package_name_moveit_config = 'mycobot_moveit_config'
    package_name_mtc = 'mycobot_mtc_pick_place_demo'

    # Package share directories
    pkg_share_moveit_config_temp = FindPackageShare(package=package_name_moveit_config)
    pkg_share_mtc_temp = FindPackageShare(package=package_name_mtc)

    # Declare launch arguments
    declare_robot_name = DeclareLaunchArgument(
        name='robot_name',
        default_value='mycobot_280',
        description='Name of the robot')

    declare_use_sim_time = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation clock')

    declare_execute = DeclareLaunchArgument(
        name='execute',
        default_value='false',
        description='Execute the planned trajectory')

    declare_use_graspnet = DeclareLaunchArgument(
        name='use_graspnet',
        default_value='true',
        description='Use GraspNet for grasp pose generation')

    declare_fallback = DeclareLaunchArgument(
        name='fallback_to_center',
        default_value='true',
        description='Fallback to center-based grasp if GraspNet fails')

    declare_max_solutions = DeclareLaunchArgument(
        name='max_solutions',
        default_value='10',
        description='Maximum number of solutions to compute')

    declare_num_grasp_candidates = DeclareLaunchArgument(
        name='num_grasp_candidates',
        default_value='5',
        description='Number of grasp pose candidates to try')

    declare_place_x = DeclareLaunchArgument(
        name='place_x',
        default_value='0.0',
        description='Place position X')

    declare_place_y = DeclareLaunchArgument(
        name='place_y',
        default_value='-0.20',
        description='Place position Y')

    declare_place_z = DeclareLaunchArgument(
        name='place_z',
        default_value='0.05',
        description='Place position Z')

    declare_launch_rviz = DeclareLaunchArgument(
        name='launch_rviz',
        default_value='false',
        description='Launch RViz with MTC panel')

    def configure_setup(context):
        """Configure MoveIt and create nodes."""
        robot_name_str = LaunchConfiguration('robot_name').perform(context)
        use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
        execute = LaunchConfiguration('execute').perform(context)
        use_graspnet = LaunchConfiguration('use_graspnet').perform(context)
        fallback = LaunchConfiguration('fallback_to_center').perform(context)
        max_solutions = LaunchConfiguration('max_solutions').perform(context)
        num_grasp_candidates = LaunchConfiguration('num_grasp_candidates').perform(context)
        place_x = LaunchConfiguration('place_x').perform(context)
        place_y = LaunchConfiguration('place_y').perform(context)
        place_z = LaunchConfiguration('place_z').perform(context)
        launch_rviz = LaunchConfiguration('launch_rviz').perform(context)

        # Get package paths
        pkg_share_moveit_config = pkg_share_moveit_config_temp.find(package_name_moveit_config)
        pkg_share_mtc = pkg_share_mtc_temp.find(package_name_mtc)

        # Config paths
        config_path = os.path.join(pkg_share_moveit_config, 'config', robot_name_str)
        mtc_config_path = os.path.join(pkg_share_mtc, 'config')
        rviz_config_path = os.path.join(pkg_share_mtc, 'rviz')

        # Config files
        initial_positions_file = os.path.join(config_path, 'initial_positions.yaml')
        joint_limits_file = os.path.join(config_path, 'joint_limits.yaml')
        kinematics_file = os.path.join(config_path, 'kinematics.yaml')
        moveit_controllers_file = os.path.join(config_path, 'moveit_controllers.yaml')
        srdf_file = os.path.join(config_path, f'{robot_name_str}.srdf')
        pilz_cartesian_limits_file = os.path.join(config_path, 'pilz_cartesian_limits.yaml')
        
        mtc_params_file = os.path.join(mtc_config_path, 'mtc_graspnet_params.yaml')
        rviz_config_file = os.path.join(rviz_config_path, 'mtc_graspnet.rviz')

        # Build MoveIt config
        moveit_config = (
            MoveItConfigsBuilder(robot_name_str, package_name=package_name_moveit_config)
            .trajectory_execution(file_path=moveit_controllers_file)
            .robot_description_semantic(file_path=srdf_file)
            .joint_limits(file_path=joint_limits_file)
            .robot_description_kinematics(file_path=kinematics_file)
            .planning_pipelines(
                pipelines=["ompl", "pilz_industrial_motion_planner", "stomp"],
                default_planning_pipeline="ompl"
            )
            .planning_scene_monitor(
                publish_robot_description=False,
                publish_robot_description_semantic=True,
                publish_planning_scene=True,
            )
            .pilz_cartesian_limits(file_path=pilz_cartesian_limits_file)
            .to_moveit_configs()
        )

        # Build place pose array
        place_pose = [float(place_x), float(place_y), float(place_z), 0.0, 0.0, 0.0]

        nodes = []

        # Create MTC node
        mtc_node = Node(
            package=package_name_mtc,
            executable="mtc_graspnet_node",
            name="mtc_graspnet_node",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {'use_sim_time': use_sim_time == 'true'},
                {'execute': execute == 'true'},
                {'use_graspnet': use_graspnet == 'true'},
                {'fallback_to_center': fallback == 'true'},
                {'max_solutions': int(max_solutions)},
                {'num_grasp_candidates': int(num_grasp_candidates)},
                {'place_pose': place_pose},
                {'start_state': {'content': initial_positions_file}},
                mtc_params_file,
            ],
        )
        nodes.append(mtc_node)

        # Optionally launch RViz with MTC panel
        if launch_rviz == 'true':
            rviz_node = Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2_mtc',
                output='screen',
                arguments=['-d', rviz_config_file] if os.path.exists(rviz_config_file) else [],
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    {'use_sim_time': use_sim_time == 'true'},
                ],
            )
            nodes.append(rviz_node)

        return nodes

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_robot_name)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_execute)
    ld.add_action(declare_use_graspnet)
    ld.add_action(declare_fallback)
    ld.add_action(declare_max_solutions)
    ld.add_action(declare_num_grasp_candidates)
    ld.add_action(declare_place_x)
    ld.add_action(declare_place_y)
    ld.add_action(declare_place_z)
    ld.add_action(declare_launch_rviz)

    # Add setup
    ld.add_action(OpaqueFunction(function=configure_setup))

    return ld
