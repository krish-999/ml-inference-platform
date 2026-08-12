# ML Inference Platform

A containerized machine learning inference platform built with Docker Compose,
TorchServe, FastAPI, Prometheus, and Grafana.

The platform demonstrates multiple independent ML use cases running through
separate application containers while sharing a centralized TorchServe
inference server.

## Requirements Covered

This project demonstrates:

1. WSL/Linux development environment
2. Internet connectivity from WSL
3. Git-based project with TorchServe inference
4. Person detection and tracking
5. Model-store and shared configuration volumes
6. Multiple ML use cases through multiple containers
7. Prometheus and Grafana monitoring

## Architecture

```text
                         Docker Compose
                              |
                         ml-platform
                              |
       +----------------------+----------------------+
       |                      |                      |
       v                      v                      v
+---------------+     +-------------------+    +-------------+
| Person        |     | Image             |    | TorchServe  |
| Tracking      |     | Classification    |    | :8080       |
| :8000         |     | :8001             |    |             |
+-------+-------+     +---------+---------+    +------+------+
        |                       |                     |
        |                       |                     |
        +-----------+-----------+                     |
                    |                                 |
                    v                                 v
              TorchServe API                    Model Store
                    |                       +------------------+
                    |                       | person_detector  |
                    |                       | image_classifier |
                    |                       +------------------+
                    |
                    +--------------------+
                                         |
                                         v
                                  Prometheus :9090
                                         |
                                         v
                                   Grafana :3000
