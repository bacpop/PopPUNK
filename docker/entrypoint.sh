#!/bin/bash --login
set -e

#echo $ROOT_PASSWD | sudo -S service ssh start

conda activate $HOME/app/env
exec "$@"