This is a demonstration of simple motion planning in TORCS simulator. All required software is packaged in a docker container.

## Requirements
 * Docker v17
 * xhost to allow run GUI application in a docker container

## Installation
 * `sudo docker build -t vaperc/torcs:latest .`

## Usage
 * connect to a docker container via ```xhost + ; sudo docker run -it -v `pwd`:/root/vaperc -v /mnt/sda4:/root/sda4 --privileged --device=/dev/snd:/dev/snd -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY vaperc/torcs:latest```
 * start torcs `torcs &`
 * turn off sound in Options->Sound
 * start new race via Race->Quick Race->New race
 * start the AI client `python snakeoil.py`
 * press Fn+F2 to change the drivers view

## Development
 * All driving logic can be extended by customizing the drive function in the snakeoil.py file.
 * Information about the available inputs can be found here: https://arxiv.org/abs/1304.1672
 * Keep in mind that some of the inputs available in the simulation won't be available on the actual car.
