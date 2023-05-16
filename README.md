# Automatic detection of new or evolving lesions in Multiple Sclerosis

## Introduction:

This repository contains code for training Deep Learning models
for the automatic segmentation of new or evolving Multiple Sclerosis (MS) lesions.
The code is an adaptation of the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) code to our specific dataset.
The dataset is composed by longitudinal FLAIR MRI images of approximately 100 people treated at
the [Hospital Clínic de Barcelona (HCB)](https://www.clinicbarcelona.org/),
together with the target masks validated by professionals from the ImaginEM research team from the HCB.

The model generated can be accessed upon a fair request.

## Using the docker image

There are two ways of building and running the Docker image,
depending on whether you want to simply use the dockerfile
or you would rather use the compose file.
The only difference is that the docker-compose.yml file contains
the configuration that without it,
you will need to tell the docker CLI.
But of course for using the Compose you need to have the file at your current directory,
which may not be always ideal.

As you will see in the following commands,
you may need to run the image with privileges,
since it needs to access the GPU.

### Building the image

#### With Docker

Solely using Docker you can build the image with the following command:

> sudo docker build -t nnunetv2 .

#### With Docker Compose

Using Docker Compose you can build the image with the following command:

> sudo docker compose build nnunetv2

### Running the image

For running the images there are two things worth considering:

1. Your input data needs to be structured in a certain way.
   Concretely, your input images need to be inside a folder called "input"
   within the directory from which you are going to run the image,
   and their names need to match the pattern `{ID}_{TIMEPOINT}.nii.gz`,
   e.g. 013_0000.nii.gz and 013_0001.nii.gz.

2. Output segmentations will be stored in the "output" folder following the pattern `mask_{ID}.nii.gz`.

#### With docker

The command for running the image with solely Docker is the following,
which is also stored inside the `predict_wo_compose.sh` bash script:

> sudo docker run --gpus 1 --entrypoint=/bin/sh --rm
> -v "${PWD}"/input:/nnunet-ms-segmentation/input/
> -v "${PWD}"/output:/nnunet-ms-segmentation/output/
> nnunetv2 predict.sh

#### With Docker Compose

With Docker Compose you may run the image with the following command:

> sudo docker compose run --rm nnunetv2