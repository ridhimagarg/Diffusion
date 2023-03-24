# ChestXNet

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
# Download pre-trained models

We have released checkpoints for the main models in the paper. Before using these models, please review the corresponding [model card](guided-diffusion/model-card.md) to understand the intended use and limitations of these models.

Here are the download links for each model trained from scratch checkpoint:

 * Model 1 diffusion [model-1.pt]()
 * Model 2 diffusion [model-2.pt]()
 * Model 3 diffusion [model-3.pt]()
 * Model 4 diffusion [model-4.pt]()
 * Model 5 diffusion [model-5.pt]()
 * Classifier [classifier.pt]
 

Here are the download links for finetuned model
 * Model 1 diffusion [model-1.pt]()

Same classifier model is used for finetuning

# Sampling from pre-trained models

To sample from these models, you can use the `classifier_sample.py`, `image_sample.py` scripts.
Here, we provide flags for sampling from all of these models.
We assume that you have downloaded the relevant model checkpoints into a folder called `models/`.

For these examples, we will generate 100 samples with batch size 4. Feel free to change these values.

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
```

## Classifier guidance

Note for these sampling runs that you can set `--classifier_scale 0` to sample from the base diffusion model.
You may also use the `image_sample.py` script instead of `classifier_sample.py` in that case.

 * Model 1 model:

```
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --class_cond True --num_heads 4 --attention_resolutions 16,8" 
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"

python scripts/classifier_sample.py --model_path {model1_path}/model090000.pt --classifier_path {classifier_path}/model085000.pt $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $CLASSIFIER_FLAGS
```

# Evaluation of diffusion model

It is presented in this [readme] (guided-diffusion/evaluations/README.md)


## Classification abormality Pleural Effusion

For this module, first we have to convert the synthetic data into the original(CheXpert) data format using this [script] (guided-diffusion/scripts/convert_generated_data_chexpert_format.py)

The code for classification is in `classifier` folder.

To use the pretrained classifier and test on the new synthetic data the following command needs to run -:

python classifier_predict.py {dataset name} densenet121-res224-all 

eg -: python classifier_predict.py chex densenet121-res224-all 

To finetune model, we used the `train.csv` generated from this [script] (guided-diffusion/scripts/convert_generated_data_chexpert_format.py) instead of `test.csv`


# Results

This table summarizes our model trained from scratch results for pleural effusion pathology:

| Dataset | FID    | sFID   |  FID (XRV) |
|---------|--------|--------| -----------|
| Model-1 | 65.95  | 79.69  |   4.87     |
| Model-2 | 446.36 | 154.64 |   10.92    |
| Model-3 | 440.38 | 309.43 |   10.03    |
| Model-4 | 105.01 | 90.80  |   5.89     |
| Model-5 | 96.24  | 105.41 |   5.27     |
| Model(18k)| 30.13 | 75.94 |   1.35     |
   

This table summarizes our model trained for multiple pathologies:

| Dataset                | FID    | sFID   |
|---------               |--------|--------|
| Atelectasis (Infected) | 109.33 | 350.12 |
| Atelectasis (Healthy)  | 118.73 | 365.16 |
| Cardiomegaly (Infected)| 105.03 | 107.95 |
| Cardiomegaly (Healthy) | 107.95 | 362.55 |



# Training models

Training diffusion models is described in the [parent repository](https://github.com/openai/guided-diffusion).