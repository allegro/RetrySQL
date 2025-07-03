export PYTHONPATH=.

.PHONY:
	install-env
	tests

install-env:
	python3 -m pip install --upgrade pip
	pip install poetry==1.8.5
	poetry install --all-extras --ansi --no-root

tests::
	poetry run python -m pytest -s --verbose

download-bird-dev-set::
	curl -O https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
	unzip dev.zip
	mv dev_20240627 ./resources/datasets/BIRD_dev
	find ./resources/datasets/BIRD_dev -type f -name "*.zip" -exec unzip -d ./resources/datasets/BIRD_dev {} \; -exec rm {} \;
	rm dev.zip

download-bird-train-set::
	curl -O https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
	unzip train.zip
	mv train ./resources/datasets/BIRD_train
	find ./resources/datasets/BIRD_train -type f -name "*.zip" -exec unzip -d ./resources/datasets/BIRD_train {} \; -exec rm {} \;
	rm train.zip
