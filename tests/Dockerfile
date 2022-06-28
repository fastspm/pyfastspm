FROM mambaorg/micromamba:latest
LABEL maintainer="Carlo Dri"
COPY --chown=$MAMBA_USER:$MAMBA_USER environment-dev.yml /tmp/environment-dev.yml

# Configure environment
ENV MPLBACKEND="agg"

# Add relevant conda packages
RUN micromamba install --yes -n base --file /tmp/environment-dev.yml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
