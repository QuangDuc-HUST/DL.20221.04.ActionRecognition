#!/bin/bash

if [ $# != 2 ]; then
echo "Usage: googledown.sh ID save_name"
exit 0


fi
curl -H "Authorization: Bearer ya29.a0AX9GBdV3hpe5-D1VQrxrlm2Je7CqzuoKYnSGPfZfVpc_Kq6gC31VhihqOOMbnO8CW4tc67FPO5INielM7wgQE7ERJdfX3Fy5kUdyfk7NWPzt0A_w5Psl80S15-i9uElSJniDv_GPd2Wm0JlmlRnpf6VWB7MJGAQaCgYKAS0SAQASFQHUCsbCWCN-F68MiNUlycaUzzuo-Q0166" https://www.googleapis.com/drive/v3/files/$1?alt=media -o $2