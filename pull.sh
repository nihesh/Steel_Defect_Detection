#!/bin/sh

rm -rf Results

scp -o ProxyCommand="ssh -W rogknastrix:22 serveo.net" -r nihesh:/home/nihesh/Documents/SteelDefectDetection/Results ./
