#!/bin/bash

# Pull from the remote server to the current directory
rsync -avz -e "ssh -i ~/.ssh/id_lichtenberg" ab67veza@lcluster1.hrz.tu-darmstadt.de:~/dev/Kairo/ .