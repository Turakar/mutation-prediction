FROM ubuntu:18.04

# install apt dependencies
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y gfortran python3.8 curl python3.8-distutils libpq-dev python3.8-dev tzdata
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# install poetry and project dependencies
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python3.8
COPY poetry.lock pyproject.toml /app/
WORKDIR /app
ENV PYTHONIOENCODING=utf-8
RUN /root/.poetry/bin/poetry env use python3.8
RUN /root/.poetry/bin/poetry install --no-dev --no-interaction

# install torch-geometric
RUN /root/.poetry/bin/poetry run pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
RUN /root/.poetry/bin/poetry run pip install torch-geometric

# install rust
RUN apt-get update && \
    apt-get install -y build-essential curl cmake && \
    curl -sSf https://sh.rustup.rs | bash -s -- -y --profile minimal
ENV PATH="$HOME/.cargo/bin:$PATH"

# copy work files
COPY . /app

# compile native module
RUN export PATH="$HOME/.cargo/bin:$PATH" && \
    cd /app/mutation_prediction_native && \
    /root/.poetry/bin/poetry run maturin develop --release

ENTRYPOINT ["/root/.poetry/bin/poetry", "run", "python"]
