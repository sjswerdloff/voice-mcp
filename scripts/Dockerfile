FROM ubuntu:25.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

RUN wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz \
    && tar -xzf piper_linux_aarch64.tar.gz -C /tmp \
    && rm piper_linux_aarch64.tar.gz \
    && mv /tmp/piper/* / && rm -rf /tmp/piper
RUN chmod +x piper

RUN wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
RUN wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

ENTRYPOINT ["./piper", "--model", "en_US-amy-medium.onnx"]
