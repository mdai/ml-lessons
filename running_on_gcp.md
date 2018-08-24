## Using Deep Learning Image for Google Cloud Engine

The Deep Learning images are VMs with pre-installed deep learning frameworks,
including TensorFlow, Keras, PyTorch, and core deep learing python packages. 
It also includes Jupyter environments for prototyping. You can choose either 
CPU or GPU images. For training, GPU images are recommended. 

You can read more [here.](https://blog.kovalevskyi.com/deep-learning-images-for-google-cloud-engine-the-definitive-guide-bc74f5fb02bc)

Before you start, you should have [GCP gcloud CLI installed on your local machine.](https://cloud.google.com/sdk/)

After gcloud CLI is setup, you can use your GCP credits to launch the Deep learning
images. For gcloud setup instructions, see [link.](https://cloud.google.com/sdk/docs/quickstarts)

## Create instance
You can change any of the following `MY_GCP_XXXX` parameters to suit your needs.
For example, `MY_GCP_ZONE` should be set to the GCP zone near you.

```sh
MY_GCP_IMAGE_FAMILY="tf-latest-cu92" # or put any required
MY_GCP_ZONE="us-east1-d"
MY_GCP_INSTANCE_NAME="gcp-instance-gpu"
MY_GCP_INSTANCE_TYPE="n1-standard-2"
MY_GCP_ACCEL="type=nvidia-tesla-k80,count=1"
gcloud compute instances create $MY_GCP_INSTANCE_NAME \
        --zone=$MY_GCP_ZONE \
        --image-family=$MY_GCP_IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator=$MY_GCP_ACCEL \
        --machine-type=$MY_GCP_INSTANCE_TYPE \
        --boot-disk-size=120GB \
        --metadata='install-nvidia-driver=True'
```

## Connect to instance
```sh
# Gcloud SSH connection and port forwarding to localhost 
# forward port 8080 for JupterLab
# forward port 6006 for tensorboard (default port)
gcloud compute ssh $MY_GCP_INSTANCE_NAME -- -L 8080:localhost:8080 -L 6006:localhost:6006
```

Now you can open your browser on your local machine on http://localhost:8080 for JupyterLab and run the jupyter notebooks.

## Clone ml-lessons repo from github on instance 
While on the GCP GCE instance, 
- Change directory to /opt/deeplearning/workspace
- Then, git clone ml-lessons repo
```sh
/opt/deeplearning/workspace
git clone https://github.com/mdai/ml-lessons.git
```

## Run Tensorboard to monitor training progress
Tensorboard is not running by default. First, find where your training output
is. For lesson3, it should be in `/opt/deeplearning/workspace/ml-lessons/logs`.

```sh
tensorboard --logdir logs --port=6006 &
```
Now, go to http://localhost:6006 for Tensorboard.

## Copy from remote instance to local 
Substitute commandline variables for actual values.

```sh
gcloud compute --project $MY_GCP_PROJECT_NAME scp --zone $MY_GCP_ZONE $MY_GCP_INSTANCE_NAME:$FILE_NAME . 
```
