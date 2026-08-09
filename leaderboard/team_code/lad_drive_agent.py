import os
import json
import random
import datetime
import pathlib
import time
import imp
from collections import deque
import math

import pandas as pd
import yaml
import cv2
import torch
import carla
import numpy as np
from PIL import Image
from easydict import EasyDict
from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner, InstructionPlanner
from team_code.pid_controller import PIDController
from timm.models import create_model
from lavis.common.registry import registry
from team_code.tools.visu import Visu

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

VEHICLE_SENSOR_LOOKUP = {
    "vehicle.lincoln.mkz2017": {"x": 1.3, "z": 2.3},
    "vehicle.volkswagen.t2": {"x": 2.0, "z": 2.5},
    "vehicle.nissan.patrol": {"x": 1.3, "z": 2.3},
    "vehicle.nissan.micra": {"x": 1.2, "z": 2.0},
    "vehicle.carlamotors.carlacola": {"x": 2.0, "z": 2.8},
}


def rotate_lidar(lidar, angle):
    radian = np.deg2rad(angle)
    return lidar @ [
        [ np.cos(radian), np.sin(radian), 0, 0],
        [-np.sin(radian), np.cos(radian), 0, 0],
        [0,0,1,0],
        [0,0,0,1]
    ]

def lidar_to_raw_features(lidar):
    def preprocess(lidar_xyzr, lidar_painted=None):

        idx = (lidar_xyzr[:,0] > -1.2)&(lidar_xyzr[:,0] < 1.2)&(lidar_xyzr[:,1]>-1.2)&(lidar_xyzr[:,1]<1.2)

        idx = np.argwhere(idx)

        if lidar_painted is None:
            return np.delete(lidar_xyzr, idx, axis=0)
        else:
            return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)

    lidar_xyzr = preprocess(lidar)

    idxs = np.arange(len(lidar_xyzr))
    np.random.shuffle(idxs)
    lidar_xyzr = lidar_xyzr[idxs]

    lidar = np.zeros((40000, 4), dtype=np.float32)
    num_points = min(40000, len(lidar_xyzr))
    lidar[:num_points,:4] = lidar_xyzr
    lidar[np.isinf(lidar)] = 0
    lidar[np.isnan(lidar)] = 0
    lidar = rotate_lidar(lidar, -90).astype(np.float32)
    return lidar, num_points

def get_entry_point():
    return "LMDriveAgent"


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class LMDriveAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):

        self._sgg = None
        self._world = None
        self._carla_client = None
        self._ego = None
        
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.active_misleading_instruction = False
        self.remaining_misleading_frames = 0

        self.visual_feature_buffer = deque(maxlen=40)
        self.ego_status_buffer = deque(maxlen=40)

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()

        self.turn_controller = PIDController(K_P=self.config.turn_KP, K_I=self.config.turn_KI, K_D=self.config.turn_KD, n=self.config.turn_n)
        self.speed_controller = PIDController(K_P=self.config.speed_KP, K_I=self.config.speed_KI, K_D=self.config.speed_KD, n=self.config.speed_n)

        model_cls = registry.get_model_class('lad_drive')

        self.traffic_light_notice = ''
        self.curr_notice = ''
        self.now_notice_frame_id = -1
        self.sample_rate = self.config.sample_rate * 2 # The frequency of CARLA simulation is 20Hz

        max_txt_len=64
            
        assert self.config.plan_anchor_path is not None, "plan anchor path must be provided for diffusion decoder"
        
        print(f"Anchor path: {self.config.plan_anchor_path}")
        print('build model...')
        model = model_cls(
            preception_model=self.config.preception_model,
            preception_model_ckpt=self.config.preception_model_ckpt,
            llm_model=self.config.llm_model,
            max_txt_len=max_txt_len,
            plan_anchor_path=self.config.plan_anchor_path,
            diffusion_hidden_dim=self.config.diffusion_hidden_dim,
        )
        self.net = model

        print('load model...')
        print('load from', self.config.lad_drive_ckpt)
        self.net.load_state_dict(torch.load(self.config.lad_drive_ckpt)["model"], strict=False)
        self.net.cuda()
        self.net.eval()

        self.softmax = torch.nn.Softmax(dim=1)
        self.prev_lidar = None
        self.prev_control = None
        self.curr_instruction = 'Drive safely.'
        self.sampled_scenarios = None
        self.instruction = ''
                
        self.save_path = None
        if SAVE_PATH is not None:
            self.save_path = pathlib.Path(os.path.join(SAVE_PATH, os.environ["BENCHMARK"], os.environ["CONFIGNAME"], os.environ["REPETITION"]))
            print("Data save path:", self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=True)
            (self.save_path / "asg_rsv").mkdir(parents=True, exist_ok=True)
            (self.save_path / "img").mkdir(parents=True, exist_ok=True)
            
            self.save_path_img_hd = self.save_path / "img_hd"
            self.visu = Visu(hd=True, save_path_img=self.save_path_img_hd)

    def _init(self):
        self._route_planner = RoutePlanner(5, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self._instruction_planner = InstructionPlanner(self.scenario_cofing_name, True)
        self.initialized = True
        random.seed(''.join([str(x[0]) for x in self._global_plan]))

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        vehicle_type = os.getenv("TEST_VEHICLE", "vehicle.lincoln.mkz2017")
        cfg = VEHICLE_SENSOR_LOOKUP.get(vehicle_type, {"x": 1.3, "z": 2.3})
        print(f"\n[INFO] Vehicle type: {vehicle_type}")
        print(f"[INFO] Using sensor x: {cfg['x']}, z: {cfg['z']}\n")

        return [
            {
                "type": "sensor.camera.rgb",
                "x": cfg["x"], #1.3 
                "y": 0.0,
                "z": cfg["z"], #2.3 
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1200,
                "height": 900,
                "fov": 100,
                "id": "rgb_front",
            },
            {
                "type": "sensor.camera.rgb",
                "x": cfg["x"], 
                "y": 0.0,
                "z": cfg["z"], 
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": cfg["x"], 
                "y": 0.0,
                "z": cfg["z"], 
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_right",
            },
            {
                "type": "sensor.camera.rgb",
                "x": -1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 180.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_rear",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": cfg["x"], 
                "y": 0.0,
                "z": cfg["z"] + 0.2, 
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -5.5, 'y': 0.0, 'z':3.5,
                'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
                # 'width': 960, 'height': 540, 'fov': 110,
                # 'width': 1280, 'height': 720, 'fov': 120,
                'width': 1920, 'height': 1080, 'fov': 110,
                'id': 'viz_third_person',
            },
            {
                "type": "sensor.camera.rgb",
                "x": -0.1, #1.3 
                "y": 0.0,
                "z": cfg["z"], #2.3 
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1200,
                "height": 900,
                "fov": 100,
                "id": "viz_front",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 6.5, 
                "y": 0.0,
                "z": 25.0, 
                "roll": 0.0,
                "pitch": -90.0,
                "yaw": 0.0,
                "width": 1920,
                "height": 1080,
                "fov": 90,
                "id": "bev",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):
        rgb_front = cv2.cvtColor(input_data["rgb_front"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(
            input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        rgb_rear = cv2.cvtColor(
            input_data["rgb_rear"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            "rgb_front": rgb_front,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            'rgb_rear': rgb_rear,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        if "viz_third_person" in input_data:
            result['viz_third_person'] = input_data['viz_third_person']
        if "viz_front" in input_data:
            result['viz_front'] = input_data['viz_front']
        if "bev" in input_data:
            result['bev'] = input_data['bev']

        pos = self._get_position(result)

        lidar_data = input_data['lidar'][1]
        result['raw_lidar'] = lidar_data

        lidar_unprocessed = lidar_data[..., :4]
        if self.prev_lidar is not None:
            lidar_unprocessed_full = np.concatenate([lidar_unprocessed, self.prev_lidar])
        else:
            lidar_unprocessed_full = lidar_unprocessed
        self.prev_lidar = lidar_unprocessed

        lidar_processed, num_points= lidar_to_raw_features(lidar_unprocessed_full)
        result['lidar'] = lidar_processed
        result['num_points'] = num_points

        result["gps"] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_waypoint"] = next_wp
        result["next_command"] = next_cmd.value
        result['measurements'] = [pos[0], pos[1], compass, speed]
        result['speed'] = speed

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point

        return result


    def update_and_collect(self, image_embeds, velocity, compass):
        self.visual_feature_buffer.append(image_embeds)
        self.ego_status_buffer.append((velocity, compass))
        buffer_list = list(self.visual_feature_buffer)
        buffer_list_ego = list(self.ego_status_buffer)
        result = buffer_list[::self.sample_rate]
        ego_statuses = buffer_list_ego[::self.sample_rate]
        if (len(buffer_list) -1) % self.sample_rate != 0:
            result.append(buffer_list[-1])
            ego_statuses.append(buffer_list_ego[-1])
        current_compass = compass
        processed_ego_statuses = []
        for v, c in ego_statuses:
            diff = c - current_compass
            status_tensor = torch.tensor([v, np.sin(diff), np.cos(diff)]).view(1,3).float().cuda()
            processed_ego_statuses.append(status_tensor)

        return torch.stack(result, 1), torch.stack(processed_ego_statuses, 1)


    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1

        tick_data = self.tick(input_data)

        if self.step < 20:
            control = carla.VehicleControl()
            control.steer = float(0)
            control.throttle = float(0)
            control.brake = float(1)
            return control

        #if self.step % 2 != 0 and self.step > 4:
        #    return self.prev_control

        velocity = tick_data["speed"]
        command = tick_data["next_command"]

        rgb_front = (
            self.rgb_front_transform(Image.fromarray(tick_data["rgb_front"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_left = (
            self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_right = (
            self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_rear = (
            self.rgb_right_transform(Image.fromarray(tick_data["rgb_rear"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_center = (
            self.rgb_center_transform(Image.fromarray(cv2.resize(tick_data["rgb_front"], (800, 600))))
            .unsqueeze(0)
            .cuda()
            .float()
        )

        last_instruction = self._instruction_planner.command2instruct(self.town_id, tick_data, self._route_planner.route)
        last_notice = self._instruction_planner.pos2notice(self.sampled_scenarios, tick_data)
        last_traffic_light_notice = self._instruction_planner.traffic_notice(tick_data)
        last_misleading_instruction = self._instruction_planner.command2mislead(self.town_id, tick_data)

        if last_notice == '':
            last_notice = last_traffic_light_notice

        if self.curr_instruction != last_instruction:
            if self.remaining_misleading_frames > 0:
                self.remaining_misleading_frames = self.remaining_misleading_frames - 1
            else:
                self.active_misleading_instruction = False
                if last_misleading_instruction!= '' and random.random() < 0.2:
                    self.curr_instruction = last_misleading_instruction
                    self.active_misleading_instruction = True
                    self.remaining_misleading_frames = 20
                else:
                    self.curr_instruction = last_instruction
                for _ in range(min(20, len(self.visual_feature_buffer))):
                    self.visual_feature_buffer.popleft()
                self.curr_notice = ''
                self.curr_notice_frame_id = -1

        input_data = {}
        input_data["rgb_front"] = rgb_front
        input_data["rgb_left"] = rgb_left
        input_data["rgb_right"] = rgb_right
        input_data["rgb_center"] = rgb_center
        input_data["rgb_rear"] = rgb_rear
        input_data['target_point'] = torch.tensor(tick_data['target_point']).cuda().view(1,2).float()
        input_data["lidar"] = (
            torch.from_numpy(tick_data["lidar"]).float().cuda().unsqueeze(0)
        )
        input_data['num_points'] = torch.tensor([tick_data['num_points']]).cuda().unsqueeze(0)
        input_data['velocity'] = torch.tensor([tick_data['speed']]).cuda().view(1, 1).float()
        input_data['text_input'] = [self.curr_instruction]
            
        image_embeds = self.net.visual_encoder(input_data)
        image_embeds, ego_statuses = self.update_and_collect(image_embeds,velocity, tick_data["compass"])
        input_data['valid_frames'] = [image_embeds.size(1)]

        if last_notice != '' and last_notice != self.curr_notice:
            new_notice_flag = True
            self.curr_notice = last_notice
            self.curr_notice_frame_id = image_embeds.size(1) - 1
        else:
            new_notice_flag = False

        with torch.cuda.amp.autocast(enabled=True):
            waypoints_pred, pred_probs, commands = self.net(input_data, inference_mode=True, image_embeds=image_embeds, ego_statuses=ego_statuses)

        mode_idx = pred_probs.argmax(dim=-1)
        mode_masks = torch.zeros(*pred_probs.shape[:2],device=pred_probs.device)
        for mask, idx in zip(mode_masks, mode_idx):
            mask[idx] = 1
        mode_masks = mode_masks.to(torch.bool)
        waypoints = waypoints_pred[mode_masks][-1]
        waypoints = waypoints.view(5, 2)
        lat_command = commands[-1]

        steer, throttle, brake, metadata = self.control_pid(waypoints, velocity)

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer) * 0.8
        control.throttle = float(throttle)
        control.brake = float(brake)
                
        if SAVE_PATH is not None:
            self.visu.visualize(img=tick_data["viz_front"][1][:, :, :3].copy(), waypoints=waypoints_pred, probs=pred_probs, prompt=input_data['text_input'][0], commands=lat_command, step=self.step)
        
        return control

    def destroy(self):
        del self.net

    def control_pid(self, waypoints, velocity):
        '''
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==5)
        waypoints = waypoints.data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if(speed < 0.01):
            angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata
