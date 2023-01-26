"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import datetime

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.gpu_util import set_gpu_use

set_gpu_use(2)

def main(no_of_images= None):
    args = create_argparser().parse_args()

    if no_of_images:
        args.num_samples = no_of_images

    print("image size", args.image_size)

    dist_util.setup_dist()
    logger.configure(dir= os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample", datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")))

    logger.log("creating model and diffusion...")

    logger.log(f"Arguments while sampling images {args}")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        logger.log(f"NUM Classes {NUM_CLASSES}")
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        logger.log(f"Classes init {classes}")
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"Classes {classes}")
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        logger.log(f"Gathered labels {gathered_labels}")
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"All labels {all_labels}")
        logger.log(f"created {len(all_images) * args.batch_size} samples")
        logger.log("All images shape:", len(all_images))

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        print("Label array", label_arr)
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")
    return out_path


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    # defaults = dict(batch_size=2, num_samples=5, timestep_respacing=250, attention_resolutions="16,8", classifier_attention_resolutions="32,16,8", class_cond=True, image_size=256, num_channels=128, num_res_blocks=3, classifier_depth=2, classifier_width=128, classifier_pool='attention', classifier_resblock_updown=True,
    # classifier_use_scale_shift_norm = True, classifier_path="/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiertrain/openai-2022-12-02-15-41-42-283455-2classes_3attempt/model085000.pt", model_path='/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/diffusiontrain/openai-2022-12-02-15-44-17-401632-2classes_3attempt/model090000.pt')
    # defaults = dict(clip_denoised=True, num_samples=5000, batch_size=8, use_ddim=False, model_path='/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/diffusiontrain/openai-2022-12-02-15-44-17-401632-2classes_3attempt/model090000.pt', classifier_path='/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiertrain/openai-2022-12-02-15-41-42-283455-2classes_3attempt/model085000.pt', classifier_scale=10.0, image_size=256, num_channels=128, num_res_blocks=3, num_heads=4, num_heads_upsample=-1, num_head_channels=-1, attention_resolutions='16,8', channel_mult='', dropout=0.0, class_cond=True, use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=False, use_fp16=False, use_new_attention_order=False, learn_sigma=False, diffusion_steps=4000, noise_schedule='linear', timestep_respacing='250', use_kl=False, predict_xstart=False, rescale_timesteps=False, rescale_learned_sigmas=False, classifier_use_fp16=False, classifier_width=128, classifier_depth=2, classifier_attention_resolutions='32,16,8', classifier_use_scale_shift_norm=True, classifier_resblock_updown=True, classifier_pool='attention')
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    defaults = dict(clip_denoised=True, num_samples=3, batch_size=3, use_ddim=False, model_path='/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/diffusiontrain/openai-2022-12-02-15-44-17-401632-2classes_3attempt/model090000.pt', classifier_path='/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiertrain/openai-2022-12-02-15-41-42-283455-2classes_3attempt/model085000.pt', classifier_scale=10.0, image_size=256, num_channels=128, num_res_blocks=3, num_heads=4, num_heads_upsample=-1, num_head_channels=-1, attention_resolutions='16,8', channel_mult='', dropout=0.0, class_cond=True, use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=False, use_fp16=False, use_new_attention_order=False, learn_sigma=False, diffusion_steps=4000, noise_schedule='linear', timestep_respacing='250', use_kl=False, predict_xstart=False, rescale_timesteps=False, rescale_learned_sigmas=False, classifier_use_fp16=False, classifier_width=128, classifier_depth=2, classifier_attention_resolutions='32,16,8', classifier_use_scale_shift_norm=True, classifier_resblock_updown=True, classifier_pool='attention')
    print("Defaults", defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
