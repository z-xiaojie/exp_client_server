#!/bin/sh

REPOSRC="https://github.com/z-xiaojie/exp_client_server.git"
LOCALREPO="/home/zxj/PyTorch-YOLOv3/data/coco/images/Crunch/src/exp_client_server"

# We do it this way so that we can abstract if from just git later on
LOCALREPO_VC_DIR=$LOCALREPO/.git

if [ ! -d $LOCALREPO_VC_DIR ]
then
    git clone $REPOSRC $LOCALREPO
else
    cd $LOCALREPO
    git pull $REPOSRC
fi

cp main.py /home/zxj/PyTorch-YOLOv3/data/coco/images/Crunch/src/main.py

# git clone https://github.com/z-xiaojie/exp_client_server.git
# sudo bash exp_client_server/update.sh
# End
