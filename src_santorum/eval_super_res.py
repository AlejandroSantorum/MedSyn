import os
import argparse
import copy
import numpy as np
from torch.optim import AdamW
from torch.utils import data
import nibabel as nib
import matplotlib.pyplot as plt

from accelerate import Accelerator

from train_super_res import Unet3D, GaussianDiffusion, Dataset, set_seed
from utils import *


# evaluator class

class Evaluator(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        model_path,
        *,
        num_eval_samples=10,
        dataset_num_max_samples=1000,  # useless for now
        dataset_seed=42,  # useless for now
        batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        save_folder='',
    ):
        super().__init__()
        self.model = diffusion_model
        self.model_path = model_path
        self.num_eval_samples = num_eval_samples
        self.dataset_num_max_samples = dataset_num_max_samples
        self.dataset_seed = dataset_seed

        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.num_frames = diffusion_model.num_frames
        self.save_folder = save_folder

        os.makedirs(self.save_folder, exist_ok=True)

        self.ds = Dataset(
            folder=folder,
            num_max_samples=self.dataset_num_max_samples,
            dataset_seed=self.dataset_seed,
            test_flag=True,
        )

        print(f'Found {len(self.ds)} 3D images at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 3D image to start evaluation'

        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True))
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999))

        self.step = 0

        self.amp = amp

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

    def load(self):
        self.accelerator.load_state(self.model_path, strict=False)

    def eval(self):
        for i in range(self.num_eval_samples):
            # sample low resolution image to be used as input
            data = next(self.dl).to(self.accelerator.device)
            img = data / 255.0  # Normalize to [0, 1]
            # sample from the model
            sampled_imgs = self.ema_model.sample(batch_size=1, img_lr=img, cond=None)

            print(f"Sampled set of images of shape {sampled_imgs.shape}", flush=True)
            print(f"Sampled image shape {sampled_imgs[0].shape}", flush=True)

            file_name = f"sample_{i}.nii.gz"
            nifti_img_path = os.path.join(self.save_folder, file_name)

            sample_img = sampled_imgs[0].squeeze(0).cpu().numpy()  # remove batch and channel dimension

            # store 3D image as nifti
            img = nib.Nifti1Image(sample_img, affine=np.eye(4))
            nib.save(img, nifti_img_path)
            print(f"Saved {nifti_img_path} (size {sample_img.shape})")

            # save sample slice as png
            sampled_slice_img = sample_img[:,:,32]  # choose slice index 32
            plt.imshow(sampled_slice_img, cmap="gray")
            plt.axis("off")
            plt.savefig(os.path.join(self.save_folder, f"sample_{i}.png"))
            plt.close()

            print(f"Sampled image shape {sampled_slice_img.shape}", flush=True)
            print(f"Min value: {sampled_slice_img.min()}, Max value: {sampled_slice_img.max()}", flush=True)


def main(args):

    model = Unet3D(
        dim=56,
        # cond_dim=768,  # used for BERT text conditioning
        dim_mults=(1, 2, 4, 8),
        channels=1, # 4, # originally 4 channels
        attn_heads=4,
        # use_bert_text_cond=False,  # used for BERT text conditioning
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=False,
        block_type='resnet',
        resnet_groups=8
    )

    # image of size (N, D, H, W) = (1, 256, 256, 256) ?
    diffusion_model = GaussianDiffusion(
        denoise_fn=model,
        image_size=256,
        num_frames=256,
        text_use_bert_cls=False,
        channels=1,  # 4, # originally 4 channels
        timesteps=1000,
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.995,
        # volume_depth=64,  # not used
        ddim_timesteps=20,
    )

    evaluator = Evaluator(
        diffusion_model=diffusion_model,
        model_path=args.model_path,
        folder=args.data_dir,
        num_eval_samples=args.dataset_num_samples,
        dataset_num_max_samples=args.dataset_num_samples,  # useless for now
        dataset_seed=args.dataset_seed,  # useless for now
        batch_size=4,
        train_lr=1e-4,
        train_num_steps=1000000,
        gradient_accumulate_every=4,
        amp=True,
        step_start_ema=10000,
        update_ema_every=10,
        save_and_sample_every=1000,
        save_folder=args.save_dir,
    )

    if args.eval_seed is not None:
        print(f"Setting train seed to {args.eval_seed} ...")
        set_seed(int(args.eval_seed))

    evaluator.load()
    evaluator.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--data_dir', type=str, required=True,
                        help='Your Evaluation DATA Path')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Your Evaluation Save Path')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Your Pretrained Model Path')
    parser.add_argument('--dataset_num_samples', type=int, default=10,
                        help='Your Evaluation DATA Number of Samples')
    parser.add_argument('--dataset_seed', type=int, default=42,
                        help='Your Evaluation DATA Seed')
    parser.add_argument('--eval_seed', type=int, default=42,
                        help='Your Evaluation Seed')

    args = parser.parse_args()

    main(args)
