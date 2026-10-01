.PHONY: install lint test train clean help

help:
	@echo "Available Makefile targets:"
	@echo "  install : Upgrade pip and install pinned dependencies"
	@echo "  lint    : Run flake8 on src/ and tests/ (max-line-length=100)"
	@echo "  test    : Run pytest suite with verbose output"
	@echo "  train   : Run MLflow training pipeline"
	@echo "  clean   : Remove bytecode, pytest cache, and temporary files"

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest -v

train:
	python -m src.train

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.py[cod]') if p.is_file()]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
