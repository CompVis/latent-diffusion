FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV LD_LIBRARY_PATH /usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR /ldm

# Create the environment:
COPY environment.yaml .
COPY ldm /ldm/ldm
COPY configs /ldm/configs
COPY scripts /ldm/scripts
COPY setup.py /ldm/
COPY notebook_helpers.py /ldm/
RUN conda env create -f environment.yaml
RUN echo "conda activate ldm" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
# The code to run when container is started:
