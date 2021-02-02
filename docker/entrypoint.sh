#!/bin/bash --login
set -e

echo $ROOT_PASSWD | sudo -S service ssh start
cp -r /$POPPUNK_DBS_LOC $HOME/$POPPUNK_DBS_LOC

conda activate $HOME/app/env
exec "$@"