##!/bin/bash
#
mkdir CodeFormer #
git clone https://github.com/sczhou/CodeFormer
#
cd CodeFormer #
	python basicsr/setup.py develop #
	python scripts/download_pretrained_models.py facelib #
	python scripts/download_pretrained_models.py CodeFormer #
cd ../ #
