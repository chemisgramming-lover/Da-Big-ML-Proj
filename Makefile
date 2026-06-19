.PHONY: help install train evaluate clean run-all

help:
	@echo "Da Big ML Project - Available Commands"
	@echo "======================================"
	@echo "make install       - Install dependencies"
	@echo "make train-custom  - Train custom CNN model"
	@echo "make train-resnet  - Train ResNet50 model"
	@echo "make train-vgg     - Train VGG16 model"
	@echo "make train-all     - Train all models"
	@echo "make evaluate      - Evaluate trained models"
	@echo "make clean         - Remove generated files"
	@echo "make run-all       - Install and train all models"

install:
	pip install -r requirements.txt

train-custom:
	python src/train.py --model custom --epochs 50 --batch_size 128

train-resnet:
	python src/train.py --model resnet50 --epochs 30 --batch_size 64

train-vgg:
	python src/train.py --model vgg16 --epochs 30 --batch_size 64

train-all:
	python src/train.py --model all --epochs 50 --batch_size 128

evaluate:
	python src/evaluate.py --compare

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache logs/*
	echo "✅ Cleaned up!"

run-all: install train-all evaluate
	@echo "🎉 Complete pipeline finished!"
