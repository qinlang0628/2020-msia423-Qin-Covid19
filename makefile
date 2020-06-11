.PHONY: clean setup
setup: data/sample data/sample/time_series_covid19_confirmed_global.csv data/sample/clean_confirmed_global.csv model/exp data/sample/training_pipeline/pred_exp model/evaluation_exp.txt

# clean up the existing files
clean:
	rm -rf data/sample/*
	rm -rf model/*

# make base directory
data/sample:
	mkdir data/sample
	mkdir data/sample/training_pipeline

# download data from 
data/sample/time_series_covid19_confirmed_global.csv: data/sample
	python3 src/acquire_data.py \
	--raw_data data/sample/time_series_covid19_confirmed_global.csv \
	--bucket nw-langqin-s3 --file_name time_series_covid19_confirmed_global.csv

# generate clean data csv
data/sample/clean_confirmed_global.csv: data/sample/time_series_covid19_confirmed_global.csv
	python3 src/clean_data.py \
	--raw_data data/sample/time_series_covid19_confirmed_global.csv \
	--clean_file data/sample/clean_confirmed_global.csv

# make training pipeline dir
data/sample/training_pipeline:
	mkdir data/sample/training_pipeline

# train model
model/exp: data/sample/clean_confirmed_global.csv data/sample/training_pipeline
	python3 src/train.py \
    --model_type exp \
    --clean_file data/sample/clean_confirmed_global.csv \
    --train data/sample/training_pipeline/train \
    --test data/sample/training_pipeline/test \
    --model_dir model

# score model
data/sample/training_pipeline/pred_exp: model/exp
	python3 src/score_model.py \
	--test data/sample/training_pipeline/test \
	--model_type exp -\
	-model_dir model \
	--pred data/sample/training_pipeline/pred

# evaluate performance
model/evaluation_exp.txt: data/sample/training_pipeline/pred_exp
	python3 src/evaluate.py \
	--model_type exp \
	--test data/sample/training_pipeline/test \
	--pred data/sample/training_pipeline/pred \
	--model_dir model
