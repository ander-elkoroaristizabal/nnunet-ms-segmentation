sudo docker run --gpus 1 --entrypoint=/bin/sh --rm \
  -v "${PWD}"/input:/nnunet-ms-segmentation/input/ \
  -v "${PWD}"/output:/nnunet-ms-segmentation/output/ \
  nnunetv2 predict.sh