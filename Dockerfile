FROM python:3.11.6 AS bdi-jupyter

# Install JupyterHub and dependencies
RUN pip3 --disable-pip-version-check install --no-cache-dir \
    notebook==7.0.6 \
    jupyterlab==4.0.8 \
    jupyterlab-server==2.25.0

# Install AlphaD3M and dependencies
ADD . /biomedical-data-integration/
WORKDIR /biomedical-data-integration/
ARG BUILD_OPTION
RUN if [ -n "$BUILD_OPTION" ]; then \
      pip3 install --no-cache-dir -e .[$BUILD_OPTION]; \
    else \
      pip3 install --no-cache-dir -e .; \
    fi

# Create a user, since we don't want to run as root
RUN useradd -m bdi
ENV HOME=/home/bdi
WORKDIR $HOME
USER bdi
COPY --chown=bdi examples /home/bdi/examples

# Huggingface text config 
ENV TOKENIZERS_PARALLELISM=false

FROM bdi-jupyter AS bdi
EXPOSE 8888
ENV SERVER_PORT 8888
ENTRYPOINT ["jupyter", "notebook","--ip=0.0.0.0","--no-browser","--NotebookApp.token=''","--NotebookApp.password=''"]
