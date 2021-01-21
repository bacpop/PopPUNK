#!/bin/bash --login
set -e

service ssh start
conda activate $HOME/app/env
exec "$@"