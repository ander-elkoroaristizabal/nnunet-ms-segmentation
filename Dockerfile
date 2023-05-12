FROM python:3.10-slim
LABEL authors="Ander Elkoroaristizabal Peleteiro"

# Update, upgrade and install package building capabilities:
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install --no-install-recommends -y build-essential

# Upgrade pip
RUN pip3 install --upgrade pip

# Timezone settings
RUN DEBIAN_FRONTEND=noninteractive apt-get install tzdata
RUN echo "Europe/Madrid" > /etc/timezone
RUN dpkg-reconfigure -f noninterative tzdata

# Working directory:
WORKDIR /nnunet-ms-segmentation

# Copy requirements
COPY requirements.txt requirements.txt

# Install packages in requirements:
RUN pip3 install -r requirements.txt

# Copy necessary files:
# nnunetv2 code:
COPY nnunetv2 nnunetv2
# nnunetv2 install script:
COPY setup.py setup.py
# Trained models:
COPY nnUNet_results/Dataset100_MSSEG/nnUNetTrainerExtremeOversamplingEarlyStoppingLowLR__nnUNetPlans__3d_fullres \
    nnUNet_results/Dataset100_MSSEG/nnUNetTrainerExtremeOversamplingEarlyStoppingLowLR__nnUNetPlans__3d_fullres

# Making necessary directories:
RUN mkdir -p data/input
RUN mkdir -p data/output

# Install nnunetv2
RUN pip3 install -e .

# Set nnunetv2 necessary paths:
ENV nnUNet_raw="./data/nnUNet_raw_data"
ENV nnUNet_preprocessed="./data/nnUNet_preprocessed_data"
ENV nnUNet_results="./nnUNet_results"

# Copy predict script:
COPY predict.sh predict.sh