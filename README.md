# NowcastNet-Rewritten

## 1. Introduction

This project is a personal reimplementation of the NowcastNet inference framework. The original research, titled "Skilful nowcasting of extreme precipitation with NowcastNet," by Yuchen Zhang, Mingsheng Long et al., was published in Nature and can be accessed at <https://www.nature.com/articles/s41586-023-06184-4>. Additionally, the original code by Yuchen Zhang is available at <https://doi.org/10.24433/CO.0832447.v1>.

> Q: Why reimplement? A: Just for learning :)

## 2. Getting Started

1. Cloning the repository:

    ```bash
    git clone https://github.com/VioletsOleander/nowcastnet-rewritten.git
    ```

2. Install the package from PyPI (**make sure `python>=3.10,<3.11`**):

    ```bash
    pip install -U nowcastnet-rewritten
    ```

    or install from local:

    ```bash
    pip install .
    ```

**Notes:**

- You may need to implement your own code to read the dataset. Sample code for reading the radar dataset is provided in the `datasets` directory.
- To ensure compatibility with this reimplementation's architecture, weights have been modified and are available for download from [Hugging Face](https://huggingface.co/VioletsOleander/nowcastnet-rewritten).

## 3. Usage

To start inference, run `inference.py` with required arguments.

To get an overview of the arguments, start with the basic command:

```bash
python inference.py -h
```

Here is an example shell script `do_inference.sh` to streamline the process. You can adjust it accordingly:

```bash
#!/bin/bash
python inference.py \
    --case_type normal \
    --device cuda:0 \
    "path_to_weights" \
    "path_to_data" \
    "path_to_result" \
```

## 4. Example Inference Result

1024 x 1024:

![Inference output at 1024×1024 resolution](docs/pictures/1024x1024.png)

512 x 512:

![Inference output at 512x512 resolution](docs/pictures/512x512.png)
