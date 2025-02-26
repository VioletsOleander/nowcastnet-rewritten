# Introduction
This project is a personal reimplementation of the NowcastNet inference framework. The original research, titled "Skilful nowcasting of extreme precipitation with NowcastNet," by Yuchen Zhang, Mingsheng Long et al., was published in Nature and can be accessed at https://www.nature.com/articles/s41586-023-06184-4. Additionally, the original code by Yuchen Zhang is available at https://doi.org/10.24433/CO.0832447.v1.

# Getting Started
Begin by clone the repository:

```bash
git clone https://github.com/VioletsOleander/nowcastnet-rewritten.git
```

Next, setup the environment:

```bash
conda create -n nowcastnet
conda activate nowcastnet
pip install -r requirements.txt
```

You may need to implementated your own code to read the dataset. Sample Code for reading the radar dataset is provided in the `datasets` directory.

To ensure compatibility with this reimplementation's architecture, weights have been modified by me and are available for download from [Hugging Face](https://huggingface.co/VioletsOleander/nowcastnet-rewritten).

# Usage
To start inference, run `inference.py` with required arguments. To get an overview of the arguments, start with the basic command:

```bash
python inference.py -h
```

Here is an example shell script `do_inference.sh` to streamline the process. You can adjust it accordingly: 

```shell
#!/bin/bash
python inference.py \
    --case_type normal \
    --device cuda:0 \
    "path_to_weights" \
    "path_to_data" \
    "path_to_result" \
```

Ensure that `do_inference.sh` has executable premissions:

```bash
chmod +x do_inference.sh
```

Then run the script using:

```bash
./do_inference.sh
```

