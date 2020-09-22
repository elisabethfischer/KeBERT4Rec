FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

COPY requirements/base.txt /workspace/requirements.txt
RUN conda install --y pip
RUN pip install -r /workspace/requirements.txt