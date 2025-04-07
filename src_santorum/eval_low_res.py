import argparse
import copy

import numpy as np
from torch.utils import data
from pathlib import Path
from torch.optim import AdamW
from dataloader import cache_transformed_text
import os


from accelerate import Accelerator

from train_low_res import Unet3D, GaussianDiffusion
from utils import *






# trainer class

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            ema_decay=0.995,
            num_frames=16,
            train_batch_size=32,
            train_lr=1e-4,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            amp=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results',
            save_folder='',
            num_sample_rows=4,
            num_sample=16,
            max_grad_norm=None
    ):
        super().__init__()
        self.model = diffusion_model
        # self.model.load_state_dict(torch.load("results/model-17.pt")['model'], strict=False)
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        self.num_frames = diffusion_model.num_frames
        self.save_folder = save_folder
        self.num_sample = num_sample

        train_files = []

        for img_dir in os.listdir(folder):
            if img_dir[-3:] == 'npy':
                train_files.append({'text': os.path.join(folder, img_dir)})

        # for img_dir in os.listdir("/ocean/projects/asc170022p/lisun/r3/results/moved_img_nii_seg_lobe_256_v2"):
        #     train_files.append({"image": os.path.join(folder, img_dir),
        #                         "lobe": os.path.join(
        #                             "/ocean/projects/asc170022p/lisun/r3/results/moved_img_nii_seg_lobe_256_v2",
        #                             img_dir),
        #                         "airway": os.path.join(
        #                             "/ocean/projects/asc170022p/lisun/r3/results/moved_img_nii_seg_airway_256_label",
        #                             img_dir),
        #                         "vessel": os.path.join(
        #                             "/ocean/projects/asc170022p/lisun/r3/results/moved_img_nii_seg_vessels_256_label",
        #                             img_dir),
        #                         'text': os.path.join("/ocean/projects/asc170022p/lisun/r3/results/text_embedding_192",
        #                                              img_dir)})

        self.ds = cache_transformed_text(train_files=train_files)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True)
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999))

        self.step = 0

        self.amp = amp
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

        if amp:
            mixed_precision = "fp16"
        else:
            mixed_precision = "fp32"

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulate_every,
            mixed_precision=mixed_precision,
        )

        self.model, self.ema_model, self.dl, self.opt, self.step = self.accelerator.prepare(
            self.model, self.ema_model, self.dl, self.opt, self.step
        )

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())


    def save(self, milestone):
        self.accelerator.save_state(str(self.results_folder / f'{milestone}_ckpt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            dirs = os.listdir(self.results_folder)
            dirs = [d for d in dirs if d.endswith("ckpt")]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[0]))
            path = dirs[-1]

        self.step = int(path.split("_")[0]) * self.save_and_sample_every + 1

        self.accelerator.load_state(os.path.join(self.results_folder, path), strict=False)

    def train(
            self,
            prob_focus_present=0.,
            focus_present_mask=None,
            log_fn=noop
    ):
        assert callable(log_fn)

        self.results_folder = os.path.join(str(self.results_folder), "given_text_ddim_eval")
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)
        for i, data in enumerate(self.dl):

            text = data["text"].squeeze(dim=1)
            text = text.to(self.accelerator.device)

            for idx in range(self.num_sample):
                with torch.no_grad():

                    file_name = data['text_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]+"_sample_"+str(idx)+".npy"

                    num_samples = self.num_sample_rows ** 2
                    batches = num_to_groups(num_samples, self.batch_size)
                    all_videos_list = list(
                        map(lambda n: self.ema_model.sample(batch_size=n, cond=text), batches))
                    all_videos_list = torch.cat(all_videos_list, dim=0)
                    np.save(os.path.join(self.save_folder, str(f'{file_name}')),
                            all_videos_list.cpu().numpy())
                #all_videos_list, all_videos_list_lobe, all_videos_list_airway, all_videos_list_vessel = all_videos_list.chunk(
                #    4, dim=1)
                #all_videos_list = torch.cat(
                #    [all_videos_list, all_videos_list_lobe, all_videos_list_airway, all_videos_list_vessel], dim=0)

                #all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                #one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)',
                #                    i=self.num_sample_rows)
                #video_path = os.path.join(self.results_folder, str(f'{file_name}.gif')).replace(".npy", "")
                #video_tensor_to_gif(one_gif, video_path)

def main(args):
    
    model = Unet3D(
        dim=160,
        cond_dim=768,
        dim_mults=(1, 2, 4, 8),
        channels=4,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8
    )

    diffusion_model = GaussianDiffusion(
        denoise_fn=model,
        image_size=64,
        num_frames=64,
        text_use_bert_cls=False,
        channels=4,
        timesteps=1000,
        loss_type='l2',
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.995,
        volume_depth=64,
        ddim_timesteps=50,
    )
    
    trainer = Trainer(
        diffusion_model=diffusion_model,
        folder=args.text_feature_folder,
        ema_decay=0.995,
        num_frames=64,
        train_batch_size=1,
        train_lr=1e-4,
        train_num_steps=1000000,
        gradient_accumulate_every=4,
        amp=True,
        step_start_ema=10000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder=args.pretrain_model_path,
        save_folder=args.save_path,
        num_sample_rows=1,
        num_sample=1,
        max_grad_norm=1.0,
    )

    trainer.load(-1)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_feature_folder', type=str)
    parser.add_argument('--pretrain_model_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    main(args)