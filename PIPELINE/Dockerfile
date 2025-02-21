# https://docs.docker.com/engine/reference/builder/
#
FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN apt-get update
RUN pip install mlflow

# Bring in changes from outside container to /tmp
# (assumes my-pytorch-modifications.patch is in same directory as Dockerfile)
COPY model_module /workspace

# Reset default working directory
WORKDIR /workspace

CMD ["python", "modeling_DL.py"]




