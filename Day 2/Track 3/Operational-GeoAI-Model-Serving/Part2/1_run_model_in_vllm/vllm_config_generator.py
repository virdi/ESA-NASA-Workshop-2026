"""
Transform a TerraTorch configuration file (YAML) into a vLLM-compatible
``config.json``. Run with ``--help`` for argument details.
"""

import yaml, json
import os
import argparse


tool_description= """
This script transforms Terratorch configuration files into a format 
compatible with vLLM.
It takes a Terratorch config file as input and generates a config.json file 
required to initialize the model within the vLLM framework.
The output file is saved in the same directory as the original configuration file.
"""

def generate_configuration(terratorch_config_file: str,data_input: str):

    vllm_config={}

    with open(terratorch_config_file) as input_stream:
        terratorch_config = yaml.safe_load(input_stream)

    vllm_config["architectures"] = ["Terratorch"]
    vllm_config["num_classes"] = 0
    vllm_config["pretrained_cfg"] = terratorch_config

    # Set default test_transform under data.init_args if missing or null
    data_init_args = vllm_config["pretrained_cfg"]["data"]["init_args"]
    if data_init_args.get("test_transform") is None:
        data_init_args["test_transform"] = [
            {
                "class_path": "albumentations.pytorch.ToTensorV2"
            }
        ]
    else:
        for entry in data_init_args["test_transform"]:
            if isinstance(entry, dict) and entry.get("class_path") == "ToTensorV2":
                entry["class_path"] = "albumentations.pytorch.ToTensorV2"

    # Drop tiled_inference_parameters when it's null, so it doesn't end up in config.json
    model_init_args = vllm_config["pretrained_cfg"]["model"]["init_args"]
    if model_init_args.get("tiled_inference_parameters") is None:
        model_init_args.pop("tiled_inference_parameters", None)

    if os.path.exists(data_input):
        vllm_config["pretrained_cfg"]['input'] = load_input_file(data_input)
    else:
        vllm_config["pretrained_cfg"]['input'] = load_input_string(data_input)

    config_dirname = os.path.dirname(terratorch_config_file)

    output_file_path = os.path.splitext(terratorch_config_file)[0]+".json"

    with open(f"{config_dirname}/config.json", 'w') as file:
        json.dump(vllm_config, file, indent=2)

    print(f"Configuration file available at the path: {config_dirname}/config.json")

def load_input_string(input):
    input_data_entries = json.loads(input)
    #loop through entries and update type and shape in place
    return input_data_entries

def load_input_file(input):
    with open(input) as f:
        input_data_entries = json.load(f)
    return input_data_entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=tool_description)
    parser.add_argument("--ttconfig", 
                        help="Terratorch model configuration file",
                        type=str)
    parser.add_argument('-i','--input', 
                        help='<Required> Input data', 
                        required=True)
    args = parser.parse_args()

    generate_configuration(args.ttconfig,args.input)
