ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:latest
FROM ${BASE_IMAGE}

# Use micromamba run to ensure python and pip in mamba-user env.
RUN micromamba run -n base python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.5.1 \
    hf_transfer==0.1.8 \
    chai_lab==0.6.1

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    DISABLE_PANDERA_IMPORT_WARNING=True

 
