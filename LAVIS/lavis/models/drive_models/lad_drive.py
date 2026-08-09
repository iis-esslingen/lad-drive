"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import os
import logging
from lavis.models.drive_models.diffusion_utils.loss_utils import LossComputer
from lavis.models.drive_models.diffusion_utils.utils import CustomTransformerDecoder, CustomTransformerDecoderLayer, SinusoidalPosEmb, gen_sineembed_for_position, linear_relu_ln, norm_odo, denorm_odo
from diffusers.schedulers import DDIMScheduler
import numpy as np

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers
import peft
from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from timm import create_model

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

@registry.register_model("lad_drive")
class LADDrive(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        preception_model="",
        preception_model_ckpt='',
        load_pretrained_preception_model=True,
        llm_model="",
        llm_pretrained_ckpt='',
        load_pretrained_llm_model=False,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        has_lora=False,
        freeze_vit=True,
        max_txt_len=128,
        freeze_decoder_of_visual_encoder=True,
        has_qformer=True,
        split_section_num_for_visual_encoder=2, # save gpu memory
        plan_anchor_path="",
        diffusion_hidden_dim=512,
    ):
        super().__init__()
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
        from transformers import AutoTokenizer
        from transformers import AutoModelForCausalLM

        self.freeze_decoder_of_visual_encoder = freeze_decoder_of_visual_encoder

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.has_qformer = has_qformer
        self.plan_anchor_path = plan_anchor_path
        self.split_section_num_for_visual_encoder = split_section_num_for_visual_encoder
        self.diffusion_hidden_dim = diffusion_hidden_dim
        
        self.has_lora = has_lora
        print(f"diffusion_hidden_dim: {diffusion_hidden_dim}")

        self.visual_encoder = create_model(preception_model) #TODO with timm
        self.ln_vision = LayerNorm(self.visual_encoder.num_features)
        if load_pretrained_preception_model:
            print(f"Load pretrained preception_model_ckpt {preception_model_ckpt}")
            pretrain_weights = torch.load(preception_model_ckpt, map_location=torch.device('cpu'))['state_dict']
            self.visual_encoder.load_state_dict(pretrain_weights, strict=True)

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                if not self.freeze_decoder_of_visual_encoder:
                    if 'deocder' not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        print("Chosen llm_model:", llm_model)

        if 'opt' in llm_model:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side='left')
            self.llm_model = OPTForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
            self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)

        if load_pretrained_llm_model and os.path.isfile(llm_pretrained_ckpt):
            print(f"Loading weights from llm_pretrained_ckpt: {llm_pretrained_ckpt}")
            ckpt = torch.load(llm_pretrained_ckpt, map_location='cpu')

            # Check if checkpoint contains 'model' dict
            if 'model' in ckpt:
                ckpt = ckpt['model']

            # Load the checkpoint into the LLM
            missing_keys, unexpected_keys = self.llm_model.load_state_dict(ckpt, strict=False)
            print(f"Loaded llm_pretrained_ckpt. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})       

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        self.llm_to_diff = nn.Sequential(nn.Linear(self.llm_model.config.hidden_size, diffusion_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(diffusion_hidden_dim, diffusion_hidden_dim))
        
        self.ego_status_fc = nn.Sequential(
            nn.Linear(3+6, diffusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(diffusion_hidden_dim, diffusion_hidden_dim),
        )
        
        plan_anchor = np.load(self.plan_anchor_path) # 20,5,2
        self.plan_anchor = nn.Parameter(
                torch.tensor(plan_anchor, dtype=torch.float32),
                requires_grad=False,
            ) # 20,6,2 # TODO 5,20,5,2
        self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(diffusion_hidden_dim, 1, 1,128*5), # TODO check
                nn.Linear(diffusion_hidden_dim, diffusion_hidden_dim),
            )
        self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(diffusion_hidden_dim),
                nn.Linear(diffusion_hidden_dim, diffusion_hidden_dim),
                nn.Mish(),
                nn.Linear(diffusion_hidden_dim, diffusion_hidden_dim),
            )
        diff_decoder_layer = CustomTransformerDecoderLayer(
                num_poses=5,
                d_model=diffusion_hidden_dim,
                d_ffn=diffusion_hidden_dim*4,
                num_head=8,
                use_ego_status=True,
            )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
        self.diffusion_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="scaled_linear",
                prediction_type="sample",
            )
            
        self.command_predictor = nn.Sequential(
            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.llm_model.config.hidden_size, 6)  # Number of command classes
        )

        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                4, self.visual_encoder.num_features
            )
            self.Qformer.resize_token_embeddings(len(self.llm_tokenizer))
            self.Qformer.cls = None

        if self.has_lora:
            loraconfig = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj","v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm_model = get_peft_model(self.llm_model, loraconfig)
            self.llm_model.print_trainable_parameters()
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        print(f"max_txt_len = {max_txt_len}")
        
        trajectory_cls_weight: float = 10.0
        trajectory_reg_weight: float = 8.0
        self.waypoints_loss = LossComputer(cls_loss_weight=trajectory_cls_weight, reg_loss_weight=trajectory_reg_weight)
        self.command_loss = torch.nn.CrossEntropyLoss()

    def concat_text_image_input(self, input_embeds, input_atts, image_embeds, image_nums, end_flag_pos_list, image_atts=None):
        '''
        attention_mask:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        '''
        input_part_targets_len = []
        llm_inputs = []
        llm_attention_mask = []
        wp_target_index = []
        bs = image_embeds.size()[0]
        for i in range(bs):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            if image_atts is None:
                bs, t, n, dim = image_embeds.size()
                llm_inputs.append(
                    torch.cat([
                        input_embeds[i][:this_input_ones],
                        image_embeds[i].view(t*n, -1),
                        input_embeds[i][this_input_ones:]
                    ])
                )
            else:
                llm_inputs.append(
                    torch.cat([
                        input_embeds[i][:this_input_ones],
                        image_embeds[i],
                        input_embeds[i][this_input_ones:]
                    ])
                )
            if image_atts is None:
                bs, t, n, dim = image_embeds.size()
                llm_attention_mask.append(
                    torch.cat([
                        input_atts[i][:this_input_ones],
                        torch.ones((image_nums[i]*n), device=image_embeds.device, dtype=torch.long),
                        torch.zeros(((t-image_nums[i])*n), device=image_embeds.device, dtype=torch.long),
                        input_atts[i][this_input_ones:]
                    ])
                )
            else:
                llm_attention_mask.append(
                    torch.cat([
                        input_atts[i][:this_input_ones],
                        image_atts[i],
                        input_atts[i][this_input_ones:]
                    ])
                )
            sub_target_index = []
            for j in end_flag_pos_list[i]:
                sub_target_index.append([i, j + this_input_ones])
            wp_target_index.extend(sub_target_index)
        llm_inputs = torch.stack(llm_inputs, 0)
        llm_attention_mask = torch.stack(llm_attention_mask, 0)
        return llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index
    
    def build_gt_commands(self, gt_commands, valid_frames):
        gt_command_list = []
        for i in range(len(valid_frames)):
            valid_frames_i = valid_frames[i]
            sequence_commands = gt_commands[i, :valid_frames_i] 
            for cmd in sequence_commands:
                cmd_val = cmd.item() - 1
                if cmd_val < 0 or cmd_val > 5:
                    logging.error(f"Invalid command value: {cmd_val}. Must be between 0 and 5.")
                gt_command_list.append(cmd_val)
        gt_command_tensor = torch.tensor(gt_command_list, device=valid_frames.device).long()
        return gt_command_tensor

    def build_gt_waypoints(self, waypoints, valid_frames):
        gt_waypoints = []
        for i in range(waypoints.size(0)):
            gt_waypoints.append(waypoints[i, :valid_frames[i]])
        gt_waypoints = torch.cat(gt_waypoints, dim=0)
        return gt_waypoints
    
    def split_data(self, samples):
        res = []
        splited_samples = {}
        split_size = samples['rgb_front'].size(0) // self.split_section_num_for_visual_encoder
        for key in ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center', 'lidar', 'num_points', 'velocity']:
            splited_samples[key] = torch.split(samples[key], split_size_or_sections=split_size, dim=0)
        for i in range(self.split_section_num_for_visual_encoder):
            new_samples = {}
            for key in ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center', 'lidar', 'num_points', 'velocity']:
                new_samples[key] = splited_samples[key][i]
            res.append(new_samples)
        return res

    def forward(self, samples, inference_mode=False, image_embeds=None, ego_statuses=None):
        if image_embeds is None: # train mode
            device = samples["rgb_front"].device
            bs = samples['rgb_front'].size(0)
            t = samples['rgb_front'].size(1)
            for key in ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'rgb_center', 'lidar', 'num_points', 'velocity']:
                shapz = samples[key].size()
                samples[key] = samples[key].view(bs*t, *shapz[2:])

            if self.freeze_decoder_of_visual_encoder:
                with torch.no_grad():
                    with self.maybe_autocast():
                        image_embeds_full = []
                        splited_samples = self.split_data(samples)
                        for i in range(self.split_section_num_for_visual_encoder):
                            image_embeds = self.visual_encoder(splited_samples[i])
                            image_embeds_full.append(image_embeds)
                        image_embeds = torch.cat(image_embeds_full, dim=0)
            else:
                with self.maybe_autocast():
                    image_embeds = self.visual_encoder(samples)
        else: # inference mode
            device = image_embeds.device
            bs = image_embeds.size(0)
            t = image_embeds.size(1)
            image_embeds = image_embeds.view(bs*t, *image_embeds.size()[2:])

        qformer_texts = samples.get("nav_input", samples["text_input"])

        image_embeds = self.ln_vision(image_embeds)
        if self.has_qformer:
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            text_Qformer = self.llm_tokenizer(
                [i for i in qformer_texts for _ in range(t)],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        image_embeds = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        image_embeds = image_embeds.view(bs, t, *image_embeds.size()[1:])
        

        image_atts = None
        end_flag_pos_list = []
        n_length = image_embeds.size(2) # token number for each frame
        for i in range(bs):
            end_flag_pos_list.append([n_length*(j+1)-1 for j in range(samples['valid_frames'][i])])

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        inputs_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)
    
        llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index = self.concat_text_image_input(
                inputs_embeds, 
                text_input_tokens.attention_mask,
                image_embeds, 
                samples['valid_frames'], 
                end_flag_pos_list, 
                image_atts)
            
        wp_target_index = torch.tensor(wp_target_index, device=device).long()

        with self.maybe_autocast():
            hidden_states = self.llm_model(
                inputs_embeds=llm_inputs,
                attention_mask=llm_attention_mask,
                return_dict=False,
            )

        predicted_command_prob = self.command_predictor(hidden_states)
        predicted_command_prob = predicted_command_prob[wp_target_index[:,0], wp_target_index[:, 1]]
        N_target = wp_target_index.size(0)
        target_hidden_states = hidden_states[wp_target_index[:,0], wp_target_index[:, 1], :].unsqueeze(1)  # [N_target, 1, hidden_dim]
        target_hidden_states = self.llm_to_diff(target_hidden_states)
        if not inference_mode:
            plan_anchor = self.plan_anchor.unsqueeze(0).repeat(N_target,1,1,1)
            odo_info_fut = norm_odo(plan_anchor)
            timesteps = torch.randint(
                0, 50,
                (N_target,), device=device
            )
            noise = torch.randn(odo_info_fut.shape, device=device)
            noisy_traj_points = self.diffusion_scheduler.add_noise(
                original_samples=odo_info_fut,
                noise=noise,
                timesteps=timesteps,
            ).float()
            noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
            noisy_traj_points = denorm_odo(noisy_traj_points)
            ego_fut_mode = noisy_traj_points.shape[1]
            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=128)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(N_target,ego_fut_mode,-1)
            # 3. embed the timesteps
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(N_target,1,-1)
            # 4. begin the stacked decoder
            predicted_command_soft = torch.softmax(predicted_command_prob, dim=1)
            ego_status_features = self.build_ego_status_features(samples['ego_statuses'], samples['valid_frames'], predicted_command_soft)
            ego_status_embed = self.ego_status_fc(ego_status_features).unsqueeze(1)
            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, target_hidden_states, time_embed, ego_status_embed)
        else:
            T_trunc = 100
            step_num = 2
            self.diffusion_scheduler.set_timesteps(T_trunc, device)
            step_ratio = T_trunc / step_num
            roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64) # 10, 0
            roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
            # 1. add truncated noise to the plan anchor
            plan_anchor = self.plan_anchor.unsqueeze(0).repeat(N_target,1,1,1)
            img = norm_odo(plan_anchor)
            noise = torch.randn(img.shape, device=device)
            start_t = int(roll_timesteps[0].item())  # e.g. 10
            trunc_timesteps = torch.full((N_target,), fill_value=start_t, device=device, dtype=torch.long)
            img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
            noisy_trajs = denorm_odo(img)
            ego_fut_mode = img.shape[1]
            
            predicted_command_soft = torch.softmax(predicted_command_prob, dim=1)
            ego_status_flat = self.build_ego_status_features(ego_statuses, samples['valid_frames'], predicted_command_soft)
            ego_status_embed = self.ego_status_fc(
                ego_status_flat
            ).unsqueeze(1)
            for k in roll_timesteps[:]:
                x_boxes = torch.clamp(img, min=-1, max=1)
                noisy_traj_points = denorm_odo(x_boxes)
                # 2. proj noisy_traj_points to the query
                traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=128)
                traj_pos_embed = traj_pos_embed.flatten(-2)
                traj_feature = self.plan_anchor_encoder(traj_pos_embed)
                traj_feature = traj_feature.view(N_target,ego_fut_mode,-1)
                timesteps = k
                if not torch.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
                elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(img.device)
                        
                # 3. embed the timesteps
                k_val = int(k.item()) if torch.is_tensor(k) else int(k)
                timesteps = torch.full((N_target,), fill_value=k_val, device=device, dtype=torch.long)
                time_embed = self.time_mlp(timesteps)
                time_embed = time_embed.view(N_target,1,-1)
                # 4. begin the stacked decoder
                poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, target_hidden_states, time_embed, ego_status_embed)
                poses_reg = poses_reg_list[-1]
                poses_cls = poses_cls_list[-1]
                x_start = poses_reg[...,:2]
                x_start = norm_odo(x_start)
                img = self.diffusion_scheduler.step(
                    model_output=x_start,
                    timestep=k_val,
                    sample=img
                ).prev_sample
            predicted_waypoints = poses_reg
            predicted_probs = poses_cls

        if inference_mode:
            return predicted_waypoints,predicted_probs, predicted_command_soft

        gt_waypoints = self.build_gt_waypoints(samples['local_future_waypoints'], samples['valid_frames'])
        gt_waypoints = gt_waypoints.view(gt_waypoints.size(0), 5, 2)
        
        waypoints_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss = self.waypoints_loss(poses_reg, poses_cls, gt_waypoints, plan_anchor)
            waypoints_loss += trajectory_loss
        
        gt_commands = self.build_gt_commands(samples['gt_commands'], samples['valid_frames'])
        command_loss = self.command_loss(predicted_command_prob, gt_commands)

        predicted_command = torch.argmax(predicted_command_prob, dim=1)
        command_acc = (predicted_command == gt_commands).float().mean().item()
        loss = waypoints_loss 
        
        if samples["global_epoch"] > 2:
            loss += command_loss
        
        return {"loss": loss, 'waypoints_loss': waypoints_loss, 'command_loss': command_loss, 'cmd_acc': command_acc}


    def build_ego_status_features(self, ego_status, valid_frames, commands):
        ego_status_list = []
        for i in range(ego_status.size(0)):
            ego_status_list.append(ego_status[i, :valid_frames[i]])
        ego_status_features = torch.cat(ego_status_list, dim=0)
        
        commands = commands.to(ego_status_features.device)
        if commands.dim() == 1:
            commands_one_hot = torch.nn.functional.one_hot(commands, num_classes=6).to(dtype=ego_status_features.dtype)
        else:
            commands_one_hot = commands.to(dtype=ego_status_features.dtype)
        
        ego_status_features = torch.cat([ego_status_features, commands_one_hot], dim=-1)
        return ego_status_features
    
    def extract_last_predictions_per_batch(self, predictions, valid_frames):
        """Extract the last prediction for each batch"""
        last_predictions = []
        current_idx = 0

        for i in range(len(valid_frames)):
            batch_frames = valid_frames[i]
            last_pred_idx = current_idx + batch_frames - 1
            last_predictions.append(predictions[last_pred_idx])
            current_idx += batch_frames
        stacked_preds = torch.stack(last_predictions)
        return stacked_preds
    
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                group_name = "vit_layer_%s" % (group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        optim_params = list(parameter_group_vars.values())
        return optim_params


    @classmethod
    def from_config(cls, cfg):
        preception_model = cfg.get("preception_model")
        preception_model_ckpt = cfg.get("preception_model_ckpt")
        load_pretrained_preception_model = cfg.get('load_pretrained_preception_model', True)
        llm_model = cfg.get("llm_model")
        llm_pretrained_ckpt = cfg.get("llm_pretrained_ckpt")
        load_pretrained_llm_model = cfg.get("load_pretrained_llm_model", False)
        img_size = cfg.get("image_size")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_lora = cfg.get("has_lora", False)

        max_txt_len = cfg.get("max_txt_len", 64)
        freeze_decoder_of_visual_encoder = cfg.get("freeze_decoder_of_visual_encoder", True)
        split_section_num_for_visual_encoder = cfg.get('split_section_num_for_visual_encoder', 2)
        
        plan_anchor_path = cfg.get("plan_anchor_path", "anchors/kmeans_anchors_lmdrive_20_modes_5_poses.npy")
        diffusion_hidden_dim = cfg.get("diffusion_hidden_dim", 512)
        
        model = cls(
            img_size=img_size,
            preception_model=preception_model,
            preception_model_ckpt=preception_model_ckpt,
            load_pretrained_preception_model=load_pretrained_preception_model,
            llm_model=llm_model,
            llm_pretrained_ckpt=llm_pretrained_ckpt,
            load_pretrained_llm_model=load_pretrained_llm_model,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            max_txt_len=max_txt_len,
            freeze_decoder_of_visual_encoder=freeze_decoder_of_visual_encoder,
            split_section_num_for_visual_encoder=split_section_num_for_visual_encoder,
            has_lora=has_lora,
            plan_anchor_path=plan_anchor_path,
            diffusion_hidden_dim=diffusion_hidden_dim
        )


        return model
