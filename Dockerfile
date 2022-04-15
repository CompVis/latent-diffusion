FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
MAINTAINER Peter Willemsen <peter@codebuffet.co>
RUN echo "Installing dependencies..." && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl wget sudo git build-essential cmake pkg-config liblzma-dev libbz2-dev zlib1g-dev libssl-dev zsh clang && \
    apt-get dist-upgrade -y && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /src/python
RUN wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tgz -O python-src.tar.gz && \
    tar xzvf python-src.tar.gz --strip-components=1 && \
    rm python-src.tar.gz && \
    ./configure --enable-optimizations --prefix=/opt/python-3.8.5 && \
    make && \
    make install && \
    rm -rf /src/python
WORKDIR / 
ENV PATH="/opt/python-3.8.5/bin:${PATH}"

RUN python3 -m pip install pip==20.3
RUN pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install numpy==1.19.2 torchvision==0.11.2 albumentations==0.4.3 opencv-python==4.1.2.30 pudb==2019.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.6.1 omegaconf==2.1.1 test-tube>=0.7.5 streamlit>=0.73.1 einops==0.3.0 torch-fidelity==0.3.0 transformers==4.3.1 -e "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers" -e "git+https://github.com/openai/CLIP.git@main#egg=clip"

RUN mkdir -p /opt/ldm_package
ADD ./setup.py /opt/ldm_package
ADD ./ldm /opt/ldm_package/ldm
ADD ./configs /opt/ldm_package/configs
RUN pip3 install -e /opt/ldm_package

WORKDIR /opt/ldm

# Add dev user
RUN useradd -ms /bin/zsh ldm-dev && \
    usermod -aG sudo ldm-dev && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ldm-dev

ENTRYPOINT ["python3"]
