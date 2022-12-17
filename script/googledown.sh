#!/bin/bash

if [ $# != 3 ]; then
echo "Usage: googledown.sh ID save_name access_token_from_oath2_background"
exit 0


fi
curl -H "Authorization: Bearer "$3 https://www.googleapis.com/drive/v3/files/$1?alt=media -o $2