FROM python:3.11.6 AS bdi-jupyter

# Install JupyterHub and dependencies
RUN pip3 --disable-pip-version-check install --no-cache-dir \
    notebook==7.0.6

# Install bdikit and dependencies
ADD . /bdikit/
WORKDIR /bdikit/
RUN pip3 install --no-cache-dir -e .

# Create a user, since we don't want to run as root
RUN useradd -m bdi
ENV HOME=/home/bdi
WORKDIR $HOME
USER bdi
COPY --chown=bdi examples /home/bdi/examples

FROM bdi-jupyter AS bdi
EXPOSE 8888
ENV SERVER_PORT 8888
ENTRYPOINT ["jupyter", "notebook","--ip=0.0.0.0","--no-browser","--NotebookApp.token=''","--NotebookApp.password=''"]
