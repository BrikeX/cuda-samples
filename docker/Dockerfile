FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update
RUN apt-get install -y git vim cmake
RUN apt-get install -y curl zip unzip
RUN apt-get install -y python3 python3-pip

# Configure git tab completion in bashrc
RUN grep -q "bash-completion/completions/git" /root/.bashrc || { \
    echo '' >> /root/.bashrc && \
    echo '# Enable git tab completion' >> /root/.bashrc && \
    echo 'if ! shopt -oq posix; then' >> /root/.bashrc && \
    echo '  # Load git completion directly' >> /root/.bashrc && \
    echo '  if [ -f /usr/share/bash-completion/completions/git ]; then' >> /root/.bashrc && \
    echo '    . /usr/share/bash-completion/completions/git' >> /root/.bashrc && \
    echo '  fi' >> /root/.bashrc && \
    echo 'fi' >> /root/.bashrc; \
}
