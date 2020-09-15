# Project based on Docker infraestructure. 

To make the base image and have the container runnig, go to tool_containerized directory and run the folowing command:
```console
sudo bash build_and_run.sh
```

If all works okay, in the consolo you will be inside the container. 
To run the tool, make shure that in the folder data/input are the inputs (.tif files) and in configs are edited too acording to the new data. 
Inside the docker, go to docker_project/tool. 
Then run the tool by typing:
```console
python3 main.py
```
After the script makes all that have to do, in the host, check the output dir inside data, and there should be the output of the inference made.
