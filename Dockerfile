FROM nvcr.io/nvidia/pytorch:23.06-py3

# Install basic packages and miscellaneous dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install python packages
COPY requirements.txt /code/requirements.txt
RUN /bin/bash -c  "cd /code && pip install --no-cache-dir -r requirements.txt"

COPY setup.py /code/setup.py
RUN /bin/bash -c  "cd /code && pip install -e ."

ENV LOCAL_MODEL_NAME_OR_PATH="/code/models"

