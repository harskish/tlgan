# Disentangling Random and Cyclic Effects in Time-Lapse Sequences
<p align="center">
<img src="docs/teaser.webp" alt="valley" width="1024"/>
</p>

> **Disentangling Random and Cyclic Effects in Time-Lapse Sequences**<br>
> Erik Härkönen<sup>1</sup>, Miika Aittala<sup>2</sup>, Tuomas Kynkäänniemi<sup>1</sup>, Samuli Laine<sup>2</sup>, Timo Aila<sup>2</sup>, Jaakko Lehtinen<sup>1,2</sup><br>
> <sup>1</sup>Aalto University, <sup>2</sup>NVIDIA<br>
> https://arxiv.org/abs/2207.01413 <br>
> https://doi.org/10.1145/3528223.3530170
>
> <p align="justify"><b>Abstract: </b><i>Time-lapse image sequences offer visually compelling insights into dynamic processes that are too slow to observe in real time. However, playing a long time-lapse sequence back as a video often results in distracting flicker due to random effects, such as weather, as well as cyclic effects, such as the day-night cycle. We introduce the problem of disentangling time-lapse sequences in a way that allows separate, after-the-fact control of overall trends, cyclic effects, and random effects in the images, and describe a technique based on data-driven generative models that achieves this goal. This enables us to ``re-render'' the sequences in ways that would not be possible with the input images alone. For example, we can stabilize a long sequence to focus on plant growth over many months, under selectable, consistent weather. <br>Our approach is based on Generative Adversarial Networks (GAN) that are conditioned with the time coordinate of the time-lapse sequence. Our architecture and training procedure are designed so that the networks learn to model random variations, such as weather, using the GAN's latent space, and to disentangle overall trends and cyclic variations by feeding the conditioning time label to the model using Fourier features with specific frequencies. <br>We show that our models are robust to defects in the training data, enabling us to amend some of the practical difficulties in capturing long time-lapse sequences, such as temporary occlusions, uneven frame spacing, and missing frames.</i></p>

## Setup
See the [setup instructions](docs/SETUP.md).

## Dataset preparation
See the [dataset preprocessing instructions](docs/PREPROC.md).

## Usage
**Training a model**<br>
First, go through the dataset preparation instructions above to produce a dataset zip.
```
# Print available options
python train.py --help

# Train TLGAN on Valley using 4 GPUs.
python train.py --outdir=~/training-runs --data=~/datasets/valley_1024x1024_2225hz.zip --gpus=4 --batch=32

# Train an unconditional StyleGAN2 on Teton using 2 GPUs.
python train.py --outdir=~/training-runs --data=~/datasets/teton_512x512_2225hz.zip --gpus=2 --batch=32 --cond=none --metrics=fid50k_full
```

**Model visualizer**<br>
The interactive model visualizer can be used to explore the effects of the conditioning inputs and the latent space.
```
# Visualize a trained model
python visualize.py path/to/model.pkl
```
The UI can be scaled with the button in the top-right corner. The UI can be made fullscreen by pressing F11.

**Grid visualizer**<br>
The input grid visualizer can be used to create 2D image grids, time-lapse images (stacked strips), and videos.
All exported files (jpg, png, mp4) contain **embedded metadata** with all UI element states.
This enables previously exported data to be loaded back into the UI via drag-and-drop.
```
# Open trained model pickle in grid visualizer
python grid_viz.py /path/to/model.pkl

# Reopen UI and load state from previously exported image
python grid_viz.py /path/to/image.png
```

**Dataset visualization**<br>
Both visualizers can display dataset frames that most closely match the current conditioning variables. Set environment variable `TLGAN_DATASET_ROOT` or pass argument `--dataset_root` to specify the directory in which datasets are stored.

## Downloads
* [Pre-trained models](https://drive.google.com/drive/folders/1ZA7Gk2OIFI2cANHEHHAm3AdWLMjJCExE?usp=sharing)
* [Supplemental material](../../releases/download/supplemental/tlgan_supplemental.zip) (zip, 303 MB)

## Known issues
* NVJPEG does not work correctly with CUDA 11.0 - 11.5 <sup>[1][nvjpeg_bug]</sup>. CPU decoding will be used instead, leading to reduced performance. Affects `preproc/process_sequence.py`, `grid_viz.py`, and `visualize.py`.

## Citation
```
@article{harkonen2022tlgan,
  author    = {Erik Härkönen and Miika Aittala and Tuomas Kynkäänniemi and Samuli Laine and Timo Aila and Jaakko Lehtinen},
  title     = {Disentangling Random and Cyclic Effects in Time-Lapse Sequences},
  journal   = {{ACM} Trans. Graph.},
  volume    = {41},
  number    = {4},
  year      = {2022},
}
```

## License

The code of this repository is based on [StyleGAN3][sg3], which is released under the [NVIDIA License](docs/LICENSE_NV.txt).<br>
All modified source files are marked separately and released under the [CC BY-NC-SA 4.0](LICENSE.txt) license.<br>
The files in `./ext` are provided under the [MIT license](https://github.com/harskish/ResizeRight/raw/master/LICENSE).<br>
The file `mutable_zipfile.py` is released under the [Python License](https://github.com/python/cpython/blob/main/LICENSE).<br>
The included Roboto Mono font is licensed under the [Apache 2.0][apache2] license.

[sg3]: https://github.com/NVlabs/stylegan3
[apache2]: https://www.apache.org/licenses/LICENSE-2.0
[nvjpeg_bug]: https://github.com/pytorch/vision/issues/4378#issuecomment-104495732
