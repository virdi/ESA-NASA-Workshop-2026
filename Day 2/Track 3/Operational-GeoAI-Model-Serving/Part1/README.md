## Before running the notebook

### Set up the Python environment

See: https://github.com/terrastackai/geospatial-studio#deployment-steps-2

Create a virtual environment and install the requirements:

```bash
pip install -r requirements.txt
```

Once you have the virtual environment, set the following environment variables:

```bash
export STUDIO_API_KEY="<your api key from the UI>"
export UI_ROUTE_URL="<STUDIO_BASE_URL>"
```

### After setup, load the required dataset, backbone, and template

#### Burn scar notebook

1. Populate Studio with the following:

   - **Dataset**: Wildfire burn scars

     Run:

     ```bash
     python populate-studio/populate-studio.py datasets
     ```

     Then select the burnscar dataset.

   - **Backbone**: Prithvi_EO_V2_300M

     Run:

     ```bash
     python populate-studio/populate-studio.py backbones
     ```

     Then select the Prithvi_EO_V2_300M backbone.

   - **Template**: Segmentation - Generic template v1 and v2 models: Segmentation

     Run:

     ```bash
     python populate-studio/populate-studio.py templates
     ```

     Then select the `Segmentation - Generic template v1 and v2 models: Segmentation` template.

#### Building detection notebook

1. Populate Studio with the following:

   - **Dataset**: Inria Building dataset

     Run:

     ```bash
     python populate-studio/populate-studio.py datasets
     ```

     Then select the Inria Building dataset.

   - **Backbone**: timm_convnext_large.fb_in22k

     Run:

     ```bash
     python populate-studio/populate-studio.py backbones
     ```

     Then select the `timm_convnext_large.fb_in22k` backbone.

   - **Template**: timm_convnext : Segmentation - Segmentation of the convnext backbone models

     Run:

     ```bash
     python populate-studio/populate-studio.py templates
     ```

     Then select the `timm_convnext : Segmentation - Segmentation of the convnext backbone models` template.