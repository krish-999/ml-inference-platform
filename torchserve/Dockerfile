FROM pytorch/torchserve:latest

COPY config/torchserve/config.properties /home/model-server/config.properties

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--ncs", "--enable-model-api", "--disable-token-auth", "--model-store", "/model-store", "--ts-config", "/home/model-server/config.properties"]
