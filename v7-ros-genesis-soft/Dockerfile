FROM ros:jazzy-ros-base@sha256:ec00a06df4f573a86e4336c4585940f7f614646ef22675ace1b3e14ee0a82cb3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ros-jazzy-rosbridge-suite=2.7.1-1noble.20261006.041103 \
    && rm -rf /var/lib/apt/lists/*

CMD ["ros2", "launch", "rosbridge_server", "rosbridge_websocket_launch.xml", "address:=0.0.0.0", "port:=9090"]
