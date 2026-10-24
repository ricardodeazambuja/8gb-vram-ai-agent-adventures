# pibench — the sandbox every benchmark run happens in.
#
# pi (the agent) runs here, never on the host: its bash/edit/write tools can only touch this
# container and the task folder mounted at /work. The model server stays on the host GPU and is
# reached over the private pibench bridge (see bench/README.md).
#
#   docker build -t pibench:1 bench/
#
FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

ARG NODE_VERSION=22.20.0
ARG PI_VERSION=1.3.0

# Node for pi (the Playwright Python image ships browsers but no Node).
RUN curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" \
      | tar -xz -C /usr/local --strip-components=1 \
 && node --version

# pi, the same version as on the host. No extensions are installed: every model gets plain pi.
RUN npm install -g "@earendil-works/pi-coding-agent@${PI_VERSION}" && pi --version

# Python tools the tasks may use; the browsers for this Playwright version are already in the image.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir playwright==1.55.0 requests beautifulsoup4 pytest

# Run as uid 1000 (the image's "ubuntu" user) so files in /work stay owned by the host user.
COPY entrypoint.sh /usr/local/bin/pibench-entry
RUN chmod 755 /usr/local/bin/pibench-entry && mkdir -p /work && chown ubuntu:ubuntu /work
USER ubuntu
ENV HOME=/home/ubuntu
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/pibench-entry"]
