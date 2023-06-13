import os
import sys
import numpy as np
import argparse
from termcolor import cprint
from vpsto import VPSTO
import time

from pybullet_planning import BASE_LINK, RED, BLUE, GREEN
from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, has_gui, set_camera, wait_for_duration, wait_if_gui, apply_alpha
from pybullet_planning import Pose, Point, Euler
from pybullet_planning import multiply, invert, get_distance
from pybullet_planning import create_obj, create_attachment, Attachment
from pybullet_planning import link_from_name, get_link_pose, get_moving_links, get_link_name, get_disabled_collisions, \
    get_body_body_disabled_collisions, has_link, are_links_adjacent
from pybullet_planning import get_num_joints, get_joint_names, get_movable_joints, set_joint_positions, joint_from_name, \
    joints_from_names, get_sample_fn, plan_joint_motion
from pybullet_planning import dump_world, set_pose
from pybullet_planning import get_collision_fn, get_floating_body_collision_fn, expand_links, create_box
from pybullet_planning import pairwise_collision, pairwise_collision_info, draw_collision_diagnosis, body_collision_info
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
UR_ROBOT_URDF = os.path.join(HERE, 'data', 'universal_robot', 'ur_description', 'urdf', 'ur5.urdf')
RFL_ROBOT_URDF = os.path.join(HERE, 'data', 'eth_rfl_robot', 'eth_rfl_description', 'urdf', 'eth_rfl.urdf')

MIT_WORKSPACE_PATH = os.path.join(HERE, 'data', 'mit_3-412_workspace', 'urdf', 'mit_3-412_workspace.urdf')
EE_PATH = os.path.join(HERE, 'data', 'dms_bar_gripper.obj')
ATTACH_OBJ_PATH = os.path.join(HERE, 'data', 'bar_attachment.obj')
OBSTACLE_OBJ_PATH = os.path.join(HERE, 'data', 'box_obstacle.obj')
DUCK_OBJ_PATH = os.path.join(HERE, 'data', 'duck.obj')
ASSEMBLY_OBJ_DIR = os.path.join(HERE, 'data', 'kathrin_assembly')

TUTORIALS = {'UR'}

def UR_setup(conf, viewer=True, robot_path=UR_ROBOT_URDF, ee_path=EE_PATH, \
    workspace_path=MIT_WORKSPACE_PATH, attach_obj_path=ATTACH_OBJ_PATH, obstacle_obj_path=OBSTACLE_OBJ_PATH):
    connect(use_gui=viewer)

    # * This is how we load a robot from a URDF, a workspace from a URDF, or simply a mesh object from an obj file
    # Notice that the pybullet uses *METER* by default, make sure you scale things properly!
    robot = load_pybullet(robot_path, fixed_base=True)
    workspace = load_pybullet(workspace_path, fixed_base=True)

    # this will print all the bodies' information in your console
    dump_world()
    
    # * adjust camera pose (optional)
    # has_gui checks if the GUI mode is enabled
    if has_gui():
        camera_base_pt = (0,0,0)
        camera_pt = np.array(camera_base_pt) + np.array([1, -0.5, 0.5])
        set_camera_pose(tuple(camera_pt), camera_base_pt)

    # * each joint of the robot are assigned with an integer in pybullet
    ik_joints = get_movable_joints(robot)
    ik_joint_names = get_joint_names(robot, ik_joints)
    cprint('Joint {} \ncorresponds to:\n{}'.format(ik_joints, ik_joint_names), 'green')
    robot_start_conf = conf
    set_joint_positions(robot, ik_joints, robot_start_conf)

    # # * attach the end effector
    # ee_link_pose = get_link_pose(robot, tool_attach_link)
    # set_pose(ee_body, ee_link_pose)
    # ee_attach = create_attachment(robot, tool_attach_link, ee_body)
    # # we need to call "assign()" to update the attachment to the current end effector pose
    # ee_attach.assign()

    # let's load a bar element (obj) and a box (pybullet primitive shape) into the world
    box_body = create_obj(obstacle_obj_path)
    block_x = 0.44
    block_y = -0.35
    block_z = 0.05
    set_pose(box_body, Pose(Point(x=block_x, y=block_y, z=block_z), Euler(yaw=np.pi/2)))

    box_body1 = create_obj(obstacle_obj_path)
    block_x = 0.3
    block_y = -0.75
    block_z = 0.05
    set_pose(box_body1, Pose(Point(x=block_x, y=block_y, z=block_z), Euler(yaw=np.pi/2)))

    box_body = [box_body, box_body1]

    cprint('We loaded a box to our scene!', 'green')


    attachments = []

    # * Let's do some collision checking
    # * specify disabled link pairs for collision checking (because they are adjacent / impossible to collide)
    # link name corresponds to the ones specified in the URDF
    # again, each robot link is assigned with an integer index in pybullet
    robot_self_collision_disabled_link_names = [('base_link', 'shoulder_link'),
        ('ee_link', 'wrist_1_link'), ('ee_link', 'wrist_2_link'),
        ('ee_link', 'wrist_3_link'), ('forearm_link', 'upper_arm_link'),
        ('forearm_link', 'wrist_1_link'), ('shoulder_link', 'upper_arm_link'),
        ('wrist_1_link', 'wrist_2_link'), ('wrist_1_link', 'wrist_3_link'),
        ('wrist_2_link', 'wrist_3_link')]
    self_collision_links = get_disabled_collisions(robot, robot_self_collision_disabled_link_names)
    extra_disabled_link_names = [('base_link', 'MIT_3412_robot_base_plate'),
                                 ('shoulder_link', 'MIT_3412_robot_base_plate')]
    extra_disabled_collisions = get_body_body_disabled_collisions(robot, workspace, extra_disabled_link_names)
    

    return robot, ik_joints, workspace, attachments, box_body, self_collision_links, extra_disabled_collisions



#####################################

def do_UR(joint_trajctory ,time_dt, q0):
    robot, ik_joints, workspace, _, box_body, self_collision_links, extra_disabled_collisions = UR_setup(q0, viewer=True)
    collision_fn = get_collision_fn(robot, ik_joints, obstacles=[workspace, box_body[0], box_body[1]], self_collisions=True,
                                            disabled_collisions=self_collision_links
                                            )
    time.sleep(2)
    itr = 0
    for conf in joint_trajctory:
        conf_cost = (collision_fn(conf, diagnosis = False))
        set_joint_positions(robot, ik_joints, conf)
        time.sleep(time_dt)
        itr+=1

    time.sleep(1)


def loss_limits(candidates):
    q_min = -5.0*np.ones(6)
    q_max = 5.5*np.ones(6)
    q = candidates['pos']
    d_min = np.maximum(np.zeros_like(q), - q + q_min)
    d_max = np.maximum(np.zeros_like(q), q - q_max)
    return np.sum(d_min > 0.0, axis=(1,2)) + np.sum(d_max > 0.0, axis=(1,2))


def joint_loss(candidates):
    costs = []
    for joint_candidates in candidates['pos']:
        diffs = np.diff(joint_candidates, axis=0)
        diff_sq = np.square(diffs)
        norm = np.sum(np.sqrt(np.sum(diff_sq, axis=1)), axis=0)
        costs.append(norm*10)
    return np.array(costs)

def loss_curvature(candidates):
    dq = candidates['vel']
    ddq = candidates['acc']
    dq_sq = np.sum(dq**2, axis=-1)
    ddq_sq = np.sum(ddq**2, axis=-1)
    dq_ddq = np.sum(dq*ddq, axis=-1) 
    return np.mean((dq_sq * ddq_sq - dq_ddq**2) / (dq_sq**3 + 1e-6), axis=-1)

def loss_collision(candidates): 
    costs = []
    q0 = np.array([-1.0581236205695661, 3.885344738206894, -2.07143761874163594, 0.7322892472673206, 4.450369729249887224, 2.0478311535710176]) # Current robot configuration
    robot, ik_joints, workspace, _, box_body, self_collision_links, extra_disabled_collisions = UR_setup(q0, viewer=False)
    collision_fn = get_collision_fn(robot, ik_joints, obstacles=[workspace, box_body[0], box_body[1]], self_collisions=True,
                                            disabled_collisions=self_collision_links
                                            )
    
    for traj in candidates['pos']:
        traj_cost = 0
        for conf in traj:
            conf_cost = (collision_fn(conf, diagnosis = False))*100
            traj_cost += conf_cost
        costs.append(traj_cost)
    costs = np.array(costs)
    disconnect()
    return costs

def loss(candidates):
    cost_limits = loss_limits(candidates)
    cost_curv = loss_curvature(candidates)
    cost_collision = loss_collision(candidates)
    joint_costs = joint_loss(candidates)
    return cost_collision + cost_curv*10 + cost_limits*10 + joint_costs

def plan_path_UR():
    vpsto = VPSTO(ndof=6)
    vpsto.opt.vel_lim = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # max. rad/s for each DoF
    vpsto.opt.acc_lim = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # max. rad/s^2 for each DoF
    vpsto.opt.max_iter = 70 #max number of iterations
    vpsto.opt.N_via = 5
    vpsto.opt.N_eval = 11
    vpsto.opt.pop_size = 100


    q0 = np.array([-1.0581236205695661, 3.885344738206894, -2.07143761874163594, 0.7322892472673206, 4.450369729249887224, 2.0478311535710176]) # Current robot configuration
    qT = np.array([3.5581236205695661, 4.685344738206894, -2.57143761874163594, 0.7322892472673206, 4.650369729249887224, 3.0478311535710176])  # Desired robot configuration

    robot, ik_joints, workspace, _, box_body, self_collision_links, extra_disabled_collisions = UR_setup(q0, viewer=True)
    set_joint_positions(robot, ik_joints, q0)
    time.sleep(3)
    set_joint_positions(robot, ik_joints, qT)
    time.sleep(3)
    disconnect()
    solution = vpsto.minimize(loss, q0, qT)
    movement_duration = solution.T_best
    joint_trajctory, _, _ = solution.get_trajectory(np.linspace(0, movement_duration, int(movement_duration*1000)))
    time_dt = (movement_duration/joint_trajctory.shape[0])
    plt.plot(solution.loss_list)
    plt.show()

    do_UR(joint_trajctory ,time_dt, q0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nv', '--noviewer', action='store_true', help='Enables the viewer during planning, default True')
    parser.add_argument('-d', '--demo', default='UR', choices=TUTORIALS, \
        help='The name of the demo')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    print('Arguments:', args)

    if args.demo == 'UR':
        plan_path_UR()
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
