# Torcs gamebot

## Links
* [Visual Torcs server github](https://github.com/giuse/vtorcs/tree/nosegfault)
* [Plib](http://plib.sourceforge.net/download.html)

## Visual Torcs server requirements
1. `sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng-dev`
2. `export CFLAGS="-fPIC" && export CPPFLAGS=$CFLAGS && export CXXFLAGS=$CFLAGS`
3. `wget http://plib.sourceforge.net/dist/plib-1.8.5.tar.gz`
4. `tar xzf plib-1.8.5.tar.gz && rm -r plib-1.8.5.tar.gz && cd plib-1.8.5`
5. `./configure ; sudo make install`

## Visual Torcs server installation
1. `wget https://github.com/giuse/vtorcs/archive/nosegfault.zip`
2. `unzip nosegfault.zip && rm nosegfault.zip && cd vtorcs-nosegfault/`
3. `./configure --prefix=$(pwd)/BUILD && make`
4. `sudo make install && make datainstall`

## Requirements
* `python 3.6.4`
* `tensorflow==1.12`
* `gym==0.14.0`
