FROM nvidia/cuda:11.0-base
RUN apt update
RUN apt install python3 python3-dev python3-pip wget build-essential git -y
RUN python3 -m pip install --upgrade pip
RUN apt-get clean -y
RUN rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

COPY . /diffusion
WORKDIR /diffusion
RUN conda env create -f environment.yaml
RUN conda init bash
RUN /bin/bash -c "source /root/.bashrc"
SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "ldm"]