from enum import Enum
import io
import math

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import textwrap
import os

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class Visu:
    def __init__(self, hd, save_path_img, bev=False):
        self.hd = hd
        self.bev = bev
        if bev:
            self.tvec = np.array([[0.0, 50.0,0.0]], np.float32)
            cam_rots = [0.0, -90.0, 0.0]
            rot_matrix = Visu.get_rotation_matrix(-cam_rots[0], -cam_rots[1], cam_rots[2])
            self.rvec = cv2.Rodrigues(rot_matrix[:3, :3])[0].flatten()
        elif hd:
            self.tvec = np.array([[0.0, 3.5, 5.5]], np.float32)
            cam_rots = [0.0, -15.0, 0.0]
            rot_matrix = Visu.get_rotation_matrix(-cam_rots[0], -cam_rots[1], cam_rots[2])
            self.rvec = cv2.Rodrigues(rot_matrix[:3, :3])[0].flatten()
        else:
            self.tvec = None
            self.rvec = None
                    
        self.save_path_img = save_path_img        
        os.makedirs(self.save_path_img, exist_ok=True)
        
    def visualize(self, img, waypoints,probs, prompt, commands, step):
        if step % 2 == 0:
            W = img.shape[1]
            H = img.shape[0]
            if self.bev:
                camera_intrinsics = np.asarray(Visu.get_camera_intrinsics(W, H, 50))
            elif self.hd:
                 camera_intrinsics = np.asarray(Visu.get_camera_intrinsics(W, H, 110))
            else:
                 camera_intrinsics = np.asarray(Visu.get_camera_intrinsics(W, H, 100))
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # draw the predicted waypoints
            image = Image.fromarray(img)

            draw = ImageDraw.Draw(image)
            multimodal = len(waypoints.shape) == 4 and probs is not None and commands is not None
            
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            if multimodal:
                # multimodal
                waypoints = waypoints[-1].squeeze(0) # [temporal_dim, num_traj, num_pts, 2] -> [num_traj, num_pts, 2]
                probs = probs[-1].squeeze(0)
            
                probs_np = probs.detach().cpu().numpy()
                sorted_indices = np.argsort(probs_np) 

                num_trajs = len(probs_np)
                if self.hd:
                    num_to_vis = num_trajs
                else:
                    num_to_vis = min(10, num_trajs)
                indices_to_vis = sorted_indices[-num_to_vis:]

                palette = [
                    (3, 4, 94),
                    (2, 62, 138),  # Top scoring
                    (0, 119, 182),
                    (0, 150, 199),
                    (0, 180, 216),
                    (72, 202, 228),
                    (144, 224, 239),
                    (173, 232, 244),
                    (202, 240, 248)
                ]

                for i, idx in enumerate(indices_to_vis):
                    waypoint = waypoints[idx]
                    
                    if i == num_to_vis - 1: # Top scoring
                        r, g, b = palette[1]
                        color = (r, g, b, 255)
                        width = 5
                        radius = 6
                    else:
                        
                        color_idx = 2 + (num_to_vis - 2 - i)
                        color_idx = min(max(color_idx, 2), len(palette) - 1)
                        
                        r, g, b = palette[color_idx]
                        color = (r, g, b, 200) 
                        width = 4
                        radius = 5

                    pred_route_numpy = waypoint.detach().cpu().numpy()
                    pred_route_numpy[:,1] *= -1
                    
                    current_traj_coords = Visu.project_points(pred_route_numpy, camera_intrinsics, tvec=self.tvec, rvec=self.rvec)
                    
                    polyline_points = [tuple(p) for p in current_traj_coords]
                    
                    if len(polyline_points) > 1:
                        overlay_draw.line(polyline_points, fill=color, width=width, joint="curve")

                    for points_2d in current_traj_coords:
                        overlay_draw.circle((points_2d[0], points_2d[1]), radius=radius, fill=color)
            else:
                #unimodal   
                pred_route_numpy = waypoints.detach().cpu().numpy()
                pred_route_numpy[:,1] *= -1
                
                current_traj_coords = Visu.project_points(pred_route_numpy, camera_intrinsics, tvec=self.tvec, rvec=self.rvec)
                
                polyline_points = [tuple(p) for p in current_traj_coords]
                
                if len(polyline_points) > 1:
                    overlay_draw.line(polyline_points, fill=(2, 62, 138, 255), width=5, joint="curve")
                for points_2d in current_traj_coords:
                    overlay_draw.circle((points_2d[0], points_2d[1]), radius=6, fill=(2, 62, 138, 255))
                    #overlay_draw.ellipse((points_2d[0]-5, points_2d[1]-5, points_2d[0]+5, points_2d[1]+5), fill=(0, 122, 255, 255))
                
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            image = Image.alpha_composite(image, overlay)
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("/arial.ttf", 45)
            except IOError:
                font = ImageFont.load_default(30)
            
            line_width = 60
            prompt = prompt
            lines = textwrap.wrap(prompt, width=line_width)
            
            text_height = 0
            max_text_width = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height += bbox[3] - bbox[1] + 10
                if text_width > max_text_width:
                    max_text_width = text_width
            padding = 20
            bg_w = max_text_width + 2 * padding
            bg_h = text_height + 2 * padding
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([(0, 0), (bg_w, bg_h)], fill=(0, 0, 0, 128))
            image = Image.alpha_composite(image.convert('RGBA'), overlay)
            draw = ImageDraw.Draw(image)
            y_text = padding
            for line in lines:
                x_text = padding
                draw.text((x_text, y_text), line, font=font, fill="white")
                bbox = draw.textbbox((0, 0), line, font=font)
                y_text += bbox[3] - bbox[1] + 10
            if multimodal:
                cmd_probs = commands.detach().cpu().numpy()
                
                fig = plt.figure(figsize=(6, 4), dpi=100)
                fig.patch.set_alpha(0.0)
                ax = plt.gca()
                ax.patch.set_alpha(0.0) 
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                ax.tick_params(axis='x', colors='white', labelsize=18, labelcolor=(1, 1, 1, 1), length=0)
                ax.set_yticks([])
                target_display_order = [
                    (RoadOption.LEFT.value, "L"),
                    (RoadOption.CHANGELANELEFT.value, "CLL"),
                    (RoadOption.STRAIGHT.value, "S"),
                    (RoadOption.LANEFOLLOW.value, "LF"),
                    (RoadOption.CHANGELANERIGHT.value, "CLR"),
                    (RoadOption.RIGHT.value, "R")
                ]
                
                # save cmd_rpobs to txt
                with open(f"{self.save_path_img}/{step}_cmd_probs.txt", "w") as f:
                    for opt_val, label in target_display_order:
                        idx = opt_val - 1
                        # label,prob
                        if 0 <= idx < len(cmd_probs):
                            f.write(f"{label},{cmd_probs[idx]:.4f}\n")
                
                ordered_probs = []
                x_labels = []
                for opt_val, label in target_display_order:
                    idx = opt_val - 1
                    if 0 <= idx < len(cmd_probs):
                        ordered_probs.append(cmd_probs[idx])
                        x_labels.append(label)
                #plt.bar(x_labels, [1.0] * len(x_labels), color=(1, 1, 1, 0.01), width=0.7)
                blue = (135/255, 206/255, 250/255, 1)
                plt.bar(x_labels, ordered_probs, color=blue, width=0.7)
                plt.ylim(0, 1.1)
                plt.xticks(rotation=0, ha='center', alpha=1.0, fontweight='bold')
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', transparent=False)
                buf.seek(0)
                hist_img = Image.open(buf).convert("RGBA")
                
                hist_w, hist_h = hist_img.size
                
                paste_x = W - hist_w - 10
                paste_y = H - hist_h - 10
                image.paste(hist_img, (paste_x, paste_y), hist_img)
                plt.close(fig)
                buf.close()

            if self.save_path_img and not self.save_path_img.exists():
                self.save_path_img.mkdir(parents=True, exist_ok=True)
                
            image.save(f"{self.save_path_img}/{step}.png")
            
    def visualize_separate(self, img, waypoints, probs, prompt, commands, step):
        if step % 2 == 0:
            W = img.shape[1]
            H = img.shape[0]
            if self.bev:
                camera_intrinsics = np.asarray(Visu.get_camera_intrinsics(W, H, 50))
            elif self.hd:
                 camera_intrinsics = np.asarray(Visu.get_camera_intrinsics(W, H, 110))
            else:
                 camera_intrinsics = np.asarray(Visu.get_camera_intrinsics(W, H, 100))
            
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # draw the predicted waypoints
            image = Image.fromarray(img)
            
            # Save raw clean image
            if self.save_path_img:
                if not self.save_path_img.exists():
                    self.save_path_img.mkdir(parents=True, exist_ok=True)
                image.save(f"{self.save_path_img}/{step}_img_clean.png")

            draw = ImageDraw.Draw(image)
            multimodal = len(waypoints.shape) == 4 and probs is not None and commands is not None
            
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            if multimodal:
                # multimodal
                waypoints = waypoints[-1].squeeze(0) # [temporal_dim, num_traj, num_pts, 2] -> [num_traj, num_pts, 2]
                probs = probs[-1].squeeze(0)
            
                probs_np = probs.detach().cpu().numpy()
                sorted_indices = np.argsort(probs_np) 

                if self.save_path_img:
                    if not self.save_path_img.exists():
                        self.save_path_img.mkdir(parents=True, exist_ok=True)
                        
                    # SAVE SORTED TRAJECTORIES AND PROBS
                    waypoints_np = waypoints.detach().cpu().numpy()
                    sorted_waypoints = waypoints_np[sorted_indices]
                    sorted_probs = probs_np[sorted_indices]
                    
                    np.savez(f"{self.save_path_img}/{step}_data.npz", 
                             waypoints=sorted_waypoints, 
                             probs=sorted_probs)

                num_trajs = len(probs_np)
                
                # Only visualize top 10 trajectories to avoid overloading
                num_to_vis = min(10, num_trajs)
                indices_to_vis = sorted_indices[-num_to_vis:]
                
                palette = [
                    (3, 4, 94),
                    (2, 62, 138),  # Top scoring
                    (0, 119, 182),
                    (0, 150, 199),
                    (0, 180, 216),
                    (72, 202, 228),
                    (144, 224, 239),
                    (173, 232, 244),
                    (202, 240, 248)
                ]
                
                for i, idx in enumerate(indices_to_vis):
                    waypoint = waypoints[idx]
                    
                    if i == num_to_vis - 1: 
                        r, g, b = palette[1]
                        color = (r, g, b, 255) 
                        width = 16
                        radius = 10
                    else: 
                        color_idx = 2 + (num_to_vis - 2 - i)
                        color_idx = min(max(color_idx, 2), len(palette) - 1)
                        r, g, b = palette[color_idx]
                        
                        color = (r, g, b, 200) 
                        width = 12
                        radius = 8

                    pred_route_numpy = waypoint.detach().cpu().numpy()
                    pred_route_numpy[:,1] *= -1
                    
                    current_traj_coords = Visu.project_points(pred_route_numpy, camera_intrinsics, tvec=self.tvec, rvec=self.rvec)
                    
                    polyline_points = [tuple(p) for p in current_traj_coords]
                    
                    if len(polyline_points) > 1:
                        overlay_draw.line(polyline_points, fill=color, width=width, joint="curve")

                    for points_2d in current_traj_coords:
                        overlay_draw.circle((points_2d[0], points_2d[1]), radius=radius, fill=color)
                        #overlay_draw.ellipse((points_2d[0]-radius, points_2d[1]-radius, points_2d[0]+radius, points_2d[1]+radius), fill=color)
            else:
                #unimodal   
                pred_route_numpy = waypoints.detach().cpu().numpy()
                pred_route_numpy[:,1] *= -1
                
                # save trajectory points to npz
                if self.save_path_img:
                    if not self.save_path_img.exists():
                        self.save_path_img.mkdir(parents=True, exist_ok=True)
                    np.savez(f"{self.save_path_img}/{step}_traj.npz", waypoints=pred_route_numpy)
                
                current_traj_coords = Visu.project_points(pred_route_numpy, camera_intrinsics, tvec=self.tvec, rvec=self.rvec)
                
                polyline_points = [tuple(p) for p in current_traj_coords]
                
                if len(polyline_points) > 1:
                    overlay_draw.line(polyline_points, fill=(2, 62, 138, 255), width=16, joint="curve")
                for points_2d in current_traj_coords:
                    overlay_draw.circle((points_2d[0], points_2d[1]), radius=10, fill=(2, 62, 138, 255))
                
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            image = Image.alpha_composite(image, overlay)
            
            # Save the image with trajectories
            if self.save_path_img and not self.save_path_img.exists():
                self.save_path_img.mkdir(parents=True, exist_ok=True)
            
            #image.save(f"{self.save_path_img}/{step}_img.png")

            # Save prompt to txt file
            with open(f"{self.save_path_img}/{step}_prompt.txt", "w") as f:
                f.write(prompt)

            # Save histogram
            if multimodal:
                cmd_probs = commands.detach().cpu().numpy()
                
                fig = plt.figure(figsize=(6, 2), dpi=100)
                fig.patch.set_alpha(1.0) # White background
                ax = plt.gca()
                ax.patch.set_alpha(1.0)
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                ax.tick_params(axis='x', colors='black', labelsize=18, labelcolor='black', length=0)
                ax.set_yticks([])
                target_display_order = [
                    (RoadOption.LEFT.value, "L"),
                    (RoadOption.CHANGELANELEFT.value, "CLL"),
                    (RoadOption.STRAIGHT.value, "S"),
                    (RoadOption.LANEFOLLOW.value, "LF"),
                    (RoadOption.CHANGELANERIGHT.value, "CLR"),
                    (RoadOption.RIGHT.value, "R")
                ]
                
                # save cmd_rpobs to txt
                with open(f"{self.save_path_img}/{step}_cmd_probs.txt", "w") as f:
                    for opt_val, label in target_display_order:
                        idx = opt_val - 1
                        # label,prob
                        if 0 <= idx < len(cmd_probs):
                            f.write(f"{label},{cmd_probs[idx]:.4f}\n")
                
                ordered_probs = []
                x_labels = []
                for opt_val, label in target_display_order:
                    idx = opt_val - 1
                    if 0 <= idx < len(cmd_probs):
                        ordered_probs.append(cmd_probs[idx])
                        x_labels.append(label)
                
                blue = (135/255, 206/255, 250/255, 1)
                plt.bar(x_labels, [1.0] * len(x_labels), color=(0.9, 0.9, 0.9, 1.0), width=0.7) 
                plt.bar(x_labels, ordered_probs, color=blue, width=0.7)
                plt.ylim(0, 1.05)
                plt.xticks(rotation=0, ha='center', alpha=1.0, fontweight='bold')
                
                plt.tight_layout()
                #plt.savefig(f"{self.save_path_img}/{step}_hist.png", format='png')
                plt.close(fig)

            
    @staticmethod
    def get_camera_intrinsics(w, h, fov):
        """
        Get camera intrinsics matrix from width, height and fov.
        Returns:
            K: A float32 tensor of shape ``[3, 3]`` containing the intrinsic calibration matrices for
            the carla camera.
        """
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0

        K = torch.tensor(K, dtype=torch.float32)
        return K
    @staticmethod
    def project_points(points2D_list, K, tvec=None, rvec=None):

        all_points_2d = []
        if rvec is None:
            rvec_new = np.zeros((3, 1), np.float32) 
        else:
            rvec_new = np.array([[-rvec[1], rvec[2], rvec[0]]], np.float32)
        if tvec is None:           
            tvec = np.array([[0.0, 2.3, 0.1]], np.float32)

        # print(f"rvec_new: {rvec_new}")
        for point in  points2D_list:
            pos_3d = np.array([point[0], 0, point[1]+tvec[0][2]])
            # Define the distortion coefficients 
            dist_coeffs = np.zeros((5, 1), np.float32) 
            points_2d, _ = cv2.projectPoints(pos_3d, 
                                rvec=rvec_new, tvec=tvec, 
                                cameraMatrix=K, 
                                distCoeffs=dist_coeffs)
            all_points_2d.append(points_2d[0][0])
                
        return all_points_2d
    @staticmethod
    def get_rotation_matrix(roll, pitch, yaw):
        roll = roll * np.pi / 180.0
        pitch = pitch * np.pi / 180.0
        yaw = yaw * np.pi / 180.0
        yawMatrix = np.matrix([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        pitchMatrix = np.matrix([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        rollMatrix = np.matrix([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        R = yawMatrix * pitchMatrix * rollMatrix
        R = pitchMatrix * yawMatrix * rollMatrix
        #inverse rotation
        R = R.T
        return R
