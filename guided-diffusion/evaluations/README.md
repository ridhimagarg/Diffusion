# Evaluations

Performed 3 ways of evaluation -: Simple (Inceptionv3 imagenet), class-wise (Inception v3 imagenet), domain specific XRV (Inception V3 DenseNet XRV) 

To compare different generative models, we use FID, sFID and XRV FID. These metrics can all be calculated using batches of samples, which we store in `.npz` (numpy) files.

# Download batches

We provide pre-computed sample batches for the reference datasets(original CheXpert dataset), our diffusion models sampled. These are all stored in `.npz` format.

Reference dataset batches contain pre-computed statistics over the whole dataset, as well as 5,000 images. All other batches contain 50,000 images which can be used to compute statistics.

Here are links to download all of the sample and reference batches:

 * CheXpert
   * CheXpert Pleural Effusion Pathology: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz)
   * CheXpert Pleural Effusion Pathology (Class labels(Infected, Healthy)): [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz)
     * [Model -1 samples (classifier scale 1.0)]()
     * [Model -5 samples (classifier scale 1.0)]()
     * [Finetuned model samples (18k steps)]()
     * [Finetuned model samples (28k steps)]()
     * [Finetuned model samples (36k steps)]()
    
  * CheXpert Multi Class
    * CheXpert All Pathology: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz)
    * [Model samples (classifier scale 1.0)]()




# Run evaluations

First, generate or download a batch of samples and download the corresponding reference batch for the given dataset. For this example, we'll use CheXpert dataset, so the refernce batch is `5000_256_256_3_reference_batch.npz` and we can use the sample batch `samples_5000x256x256x3.npz`.

Next, run the `fid_score_inception.py` or `fid_score_calsswise_inception.py` or `fid_score_densenet.py` script. Pass two arguments to the script: the reference batch and the sample batch. The script will download the InceptionV3 model or Inception V3(trained on DenseNet XRV) used for evaluations into the current working directory (if it is not already present).

The output of the script will look something like this, where the first `...` is a bunch of verbose TensorFlow logging:

```
$ python fid_score_inception.py 5000_256_256_3_reference_batch.npz samples_5000x256x256x3.npz
...
computing reference batch activations...
computing/reading reference batch statistics...
computing sample batch activations...
computing/reading sample batch statistics...
Computing evaluations...
Inception Score: 215.8370361328125
FID: 3.9425574129223264
sFID: 6.140433703346162
Precision: 0.8265
Recall: 0.5309
```
