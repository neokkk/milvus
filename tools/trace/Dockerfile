FROM ubuntu:22.04
 
WORKDIR /milvus

# COPY ./entrypoint.sh /milvus

# RUN chmod +x /milvus/entrypoint.sh

RUN apt update && apt install -y uuid-dev libgomp1 libaio-dev libopenblas-dev libgoogle-perftools-dev

CMD ["tail", "-f", "/dev/null"]
