# SceneMotifCoder

### SceneMotifCoder: Example-driven Visual Program Learning for Generating 3D Object Arrangements

[Hou In Ivan Tam](https://iv-t.github.io/), [Hou In Derek Pun](https://houip.github.io/), [Austin T. Wang](https://atwang16.github.io/), [Angel X. Chang](https://angelxuanchang.github.io/), [Manolis Savva](https://msavva.github.io/)

<img src="docs/static/images/teaser.webp" alt="teaser" style="width:100%"/>

[Page](https://3dlg-hcvc.github.io/smc/) | [Paper]() | [Data](https://github.com/3dlg-hcvc/smc/releases/download/v1.0/examples.zip)

## Setup Environment
We recommend using `mamba` to manage the environment.
`mamba` is a drop-in replacement for `conda` that is significantly faster and better at solving dependencies.
Run the following commands to create and activate the environment.
Replace `mamba` with `conda` in the following commands if you use `conda`.
```bash
# Create and activate the environment
mamba env create -f environment.yml
mamba activate smc
```
Create a `.env` file in the root directory of the project and add your OpenAI API key as follows:
```bash
# Inside .env
OPENAI_API_KEY=<YOUR_API_KEY>
```

## Download Data
### Example Arrangements
Download the example arrangements [here](https://1sfu-my.sharepoint.com/:u:/g/personal/hit_sfu_ca/ES9-av6IP4pBkyCeFqX_-SUByIKkNpo8nOO1-z2C3pfSlg) and extract the contents to the root of the project.

### Assets for Retrieval
SMC retrieves 3D models from the [Habitat Synthetic Scenes Dataset (HSSD)](https://3dlg-hcvc.github.io/hssd/).
To download the dataset, accept the terms and conditions of the dataset on Hugging Face ([here](https://huggingface.co/datasets/hssd/hssd-models)).
Then, clone the dataset repository (~72GB) at the root of the project:
```bash
cd smc
git lfs install
git clone https://huggingface.co/datasets/hssd/hssd-models
```
Lastly, download the asset metadata `.csv` file [here](https://huggingface.co/datasets/hssd/hssd-hab/tree/main/semantics) and place it inside the `hssd-models` directory.

### Directory Structure
You should now have the following directory structure:
```
smc
├── examples
│   ├── a_stack_of_seven_plates.glb
│   ├── ...
├── hssd-models
│   ├── semantics_objects.csv
│   ├── ...
|── ...
```

## Learn Meta-Program from Example
Run the following command to learn a meta-program from an example arrangement:
```bash
python learn.py --file examples/a_stack_of_seven_plates.glb --desc "a stack of seven plates"
```
The motif program and meta-program will be saved in `libraries/` under the corresponding directories.

To improve a meta-program with more examples, simply run the command again with a different example arrangement of the same motif type.
SMC will automatically update the meta-program using the new example.

## Generate New Arrangement
After learning a meta-program, you can use it to generate new arrangements by running the following command:
```bash
python inference.py --desc "a stack of four books"
```
By default, the generated arrangement will be saved under `outputs/`. See `inference.py` for more options.

## Citation
Please cite our work if you find it helpful:
```
@article{??????????,
    author        = {Tam, Hou In Ivan and Pun, Hou In Derek and Wang, Austin T. and Chang, Angel X. and Savva, Manolis},
    title         = {{SceneMotifCoder: Example-driven Visual Program Learning for Generating 3D Object Arrangements}},
    year          = {2024},
    eprint        = {0000.00000},
    archivePrefix = {arXiv}
}
```

## Acknowledgements
This work was funded in part by a CIFAR AI Chair, a Canada Research Chair, NSERC Discovery Grant, NSF award #2016532, and enabled by support from WestGrid and Compute Canada.
We thank Qirui Wu, Jiayi Liu, and Han-Hung Lee for helpful discussions and feedback.
