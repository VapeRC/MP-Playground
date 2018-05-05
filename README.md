# MP-Playground
Repository for experimenting with motion planning algorithms.

*NOTE*: code submitted to this repository does not go through code reviews and is not tested, so don't use it on the car itself. Instead copy the parts you need to the Robocar repo.


## Torcs
Torcs is a car racing simulator used to test motion planning algorithms. Folder also includes a simple heuristic for motion control.

### Requirements
 * Docker v17
 * xhost to allow run GUI application in a docker container

### Installation
 * `sudo docker build -t vaperc/torcs:latest .`

### Usage
 * connect to a docker container via ```xhost + ; sudo docker run -it -v `pwd`:/root/vaperc -v /mnt/sda4:/root/sda4 --privileged --device=/dev/snd:/dev/snd -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY vaperc/torcs:latest```
 * start torcs `torcs &`
 * turn off sound in Options->Sound
 * start new race via Race->Quick Race->New race
 * start the AI client `python snakeoil.py`
 * press Fn+F2 to change the drivers view


## DDPG

Motion planning and control using deep reinforcement learning. Based on (https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html).

### Usage
  * Models are stored in the `data` folder.
  * To run simulation first start torcs and select quick race
  * Run `python3 ddpg_torcs.py train` to train the agent
  * Run `python3 ddpg_torcs.py run` to simply evaluate the agent
  * See `python3 ddpg_torcs.py -h` for more options. 
