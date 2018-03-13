docker run -it --rm -v $(pwd)/hyperdock_config:/app/hyperdock/config:ro erikgartner/hyperdock-supervisor:latest --name deep --image erikgartner/wasp-cifar --config_module deep --trials 26 --mongo mongo://172.17.0.1:27017/hyperdock/jobs
