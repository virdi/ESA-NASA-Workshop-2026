# GEOStudio Workshop Guide

This guide provides comprehensive instructions for setting up and running the GEOStudio workshop, covering both administrator setup and participant usage.

## Table of Contents

- [For Participants](#for-participants)
  - [Getting Started](#getting-started)
  - [What You'll Learn](#what-youll-learn)
  - [Workshop Labs](#workshop-labs)
- [For System Administrators](#for-system-administrators)
  - [Deployment](#deployment)
  - [Loading Required Assets](#loading-required-assets)
  - [User Account Setup](#user-account-setup)

---

## For Participants

### Getting Started

If you're a workshop participant, your system administrator has already set up your GEOStudio account. Follow these steps to get started:

#### 1. Obtain Your Credentials

Your system administrator will provide you with:
- Studio UI URL
- Username
- Initial password (if applicable)

#### 2. Create Your API Key

1. Navigate to the Studio UI using the provided URL
2. Log in with your credentials
3. Click on **"Manage your API keys"** in the user menu
4. Click **"Generate a new API key"**
5. **Important**: Copy and save your API key securely
6. Store it in a safe location (e.g., password manager or secure notes)

> **Note**: Your API key is required for all programmatic interactions with the platform through the workshop notebooks.

### What You'll Learn

By completing this workshop, you will gain hands-on experience with:

- **Platform Setup**: Connecting to and authenticating with the GEOStudio platform
- **Model Fine-Tuning**: Customizing pre-trained foundation models for specific geospatial tasks
  - Burn scar detection from satellite imagery
  - Building detection and segmentation
- **Dataset Management**: Working with curated geospatial datasets
- **Model Evaluation**: Testing your fine-tuned models on new data
- **Results Visualization**: Analyzing and visualizing model outputs in the GEOStudio UI
- **Model Comparison**: Comparing different models and training configurations (advanced)
- **Model Export**: Preparing models for production deployment with vLLM

### Workshop Labs

Complete the following labs in order:

1. **[Wildfire Burn Scar Detection Lab](./lab1-esa-nasa-burnscars-workflow.ipynb)**
   - Fine-tune a model to detect wildfire burn scars from satellite imagery
   - Use the Prithvi_EO_V2_300M foundation model
   - Work with the Wildfire burn scars dataset

2. **[Building Detection Lab](./lab2-esa-nasa-buildings-workflow.ipynb)**
   - Train a model for building detection and segmentation
   - Use the timm_convnext_large.fb_in22k backbone
   - Work with the Inria Building dataset

---

## For System Administrators

### Deployment

#### Prerequisites

- Kubernetes cluster (or compatible environment)
- Sufficient compute resources (CPU, GPU, memory, storage)
- Administrative access to the cluster

#### Deployment Steps

Follow the comprehensive deployment guide:

📖 **[Deployment Documentation](https://terrastackai.github.io/geospatial-studio/workshop/prework/)**

This guide covers:
- Infrastructure requirements
- Installation procedures
- Configuration options
- Post-deployment verification

### Loading Required Assets

After successful deployment, populate the platform with the necessary datasets, backbones, and templates for the workshop.

#### Environment Setup

1. **Create Python Virtual Environment**

   See detailed instructions: [Deployment Steps](https://github.com/terrastackai/geospatial-studio#deployment-steps-2)

   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**

   **Important**: Use the system API key to make assets available globally to all users.

   ```bash
   export STUDIO_API_KEY="<system-admin-api-key>"
   export UI_ROUTE_URL="<STUDIO_BASE_URL>"
   ```

   > **Tip**: You can obtain the system API key from the Studio UI from the .env file in the deployment workspace or from geofm-gateway-secrets in the cluster.

#### Asset Loading

##### For Burn Scar Detection Lab

Load the following assets in order:

1. **Dataset: Wildfire Burn Scars**

   ```bash
   python populate-studio/populate-studio.py datasets
   ```

   When prompted, select: **Wildfire burn scars**

2. **Backbone: Prithvi_EO_V2_300M**

   ```bash
   python populate-studio/populate-studio.py backbones
   ```

   When prompted, select: **Prithvi_EO_V2_300M**

3. **Template: Segmentation - Generic template v1 and v2**

   ```bash
   python populate-studio/populate-studio.py templates
   ```

   When prompted, select: **Segmentation - Generic template v1 and v2 models: Segmentation**

##### For Building Detection Lab

Load the following assets in order:

1. **Dataset: Inria Building Dataset**

   ```bash
   python populate-studio/populate-studio.py datasets
   ```

   When prompted, select: **Inria Building dataset**

2. **Backbone: timm_convnext_large.fb_in22k**

   ```bash
   python populate-studio/populate-studio.py backbones
   ```

   When prompted, select: **timm_convnext_large.fb_in22k**

3. **Template: timm_convnext Segmentation**

   ```bash
   python populate-studio/populate-studio.py templates
   ```

   When prompted, select: **timm_convnext : Segmentation - Segmentation of the convnext backbone models**

#### Verification

After loading assets, verify they appear correctly in the Studio UI:

1. Log in to the Studio UI as an admin
2. Navigate to:
   - **Dataset Factory** section - confirm both datasets are visible
   - **Fine-tuning Studio (Base Model)** section - confirm both backbones are available
3. **Templates** section - confirm both templates are loaded
4. Check that assets are created by the system user with a default (Studio.Dev@dev.com)
5. These assets should be visible to all users

### User Account Setup

#### Option 1: Keycloak Admin Console (Recommended)

1. Access the Keycloak admin console
2. Navigate to the `geostudio` realm
3. Create user accounts for each participant:
   - Set username (e.g., `participant1`, `participant2`)
   - Set email address
   - Enable the account
   - Set initial password (or require password reset on first login)
   - Set email verification to false (optional)

#### Option 2: Automated User Creation Script

You can create a script to automate user creation via the Keycloak API.

#### Post-Creation Steps

1. **Communicate Credentials**: Securely share login credentials with participants
2. **Provide Access Information**: Send participants:
   - Studio UI URL
   - Username
   - Initial password (if applicable)
   - Link to this README for getting started instructions
3. **Verify Access**: Test login with a sample account to ensure everything works

---

## Troubleshooting

### Common Issues

**Issue**: API key not working
- **Solution**: Ensure you copied the entire key including the `pak-` prefix
- **Solution**: Verify the key hasn't expired (check in UI)

**Issue**: Assets not visible to participants
- **Solution**: Ensure assets were loaded using the system admin API key
- **Solution**: Check asset sharing/visibility settings in the UI

**Issue**: Model training fails
- **Solution**: Verify sufficient GPU resources are available
- **Solution**: Check logs in the Studio UI for specific error messages

**Issue**: Cannot access Studio UI
- **Solution**: Verify the URL is correct and the service is running
- **Solution**: Check network connectivity and firewall rules

### Getting Help

- Check the [official documentation](https://terrastackai.github.io/geospatial-studio/)
- Review deployment logs for error messages
- Contact your system administrator for access issues
- Consult the workshop facilitator during the session

---

## Additional Resources

- **Documentation**: [GEOStudio Docs](https://terrastackai.github.io/geospatial-studio/)
- **GitHub Repository**: [geospatial-studio](https://github.com/terrastackai/geospatial-studio)
- **Workshop Materials**: Additional notebooks and resources in the `tasks/` directory

