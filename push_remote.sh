#!/bin/bash

# Push the current directory to the remote server
rsync -avz \
 --exclude ".git" \
 --exclude ".vscode" \
 --exclude ".github" \
 --exclude "frontend" \
 --include "sumo_data/" \
 --include "sumo_data/static_features_cache.json" \
 --include "sumo_data/network.net.xml" \
 --exclude "sumo_data/*" \
 -e "ssh -i ~/.ssh/id_dgx" \
 . dgx:~/dev/Kairo/