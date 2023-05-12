sudo docker run --gpus 1 --entrypoint=/bin/sh --rm \
  -v "${PWD}"/data/input:/nnunet-ms-segmentation/data/input/ \
  -v "${PWD}"/data/output:/nnunet-ms-segmentation/data/output/ \
  nnunetv2 predict.sh