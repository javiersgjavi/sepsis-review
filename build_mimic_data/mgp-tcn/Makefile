MAKEFLAGS = --warn-undefined-variables

CONFIG_FILES_PATH = configs
EXPERIMENTS_PATH = experiments

CONFIG_FILES = $(wildcard $(CONFIG_FILES_PATH)/*.json)
CONFIGS = $(foreach FILE,$(CONFIG_FILES),$(basename $(notdir $(FILE))))

EXPERIMENT_TARGETS = $(foreach CONFIG,$(CONFIGS),$(EXPERIMENTS_PATH)/$(CONFIG))

MYSQL_DATABASE := mimic
MYSQL_USER := postgres


.PHONY = run_experiments query

build_docker:
	docker build . -t sepsis:v2
	docker run -it --gpus all -p 5432:5432 sepsis:v2

run_docker:
	docker run -it --gpus all -p 5432:5432 sepsis:v2

generate_data:
	python3 -m src.mgp_tcn.mgp_tcn_fit with configs/paper_configs/config_split0.json
	echo 'Ignore the last message, the process has been succesfully completed'
	echo 'Done!'
	 
all: query run_experiments

mock_data_results:
	python3 -m src.mgp_tcn.test_mgp_tcn -F $@
	@mv $@/1/* $@
	@rm -r $@/1

query:
	echo Runnung query...
	cd src/query && ./main.sh $(MYSQL_DATABASE) $(MYSQL_USER)

run_experiments: $(EXPERIMENT_TARGETS)
	echo $(CONFIG_FILES)
	echo $(CONFIGS)
	echo 'asdf'
	
$(EXPERIMENTS_PATH)/%: $(CONFIG_FILES_PATH)/%.json
	echo 'daf'
	@echo Running config file $<
	@-mkdir -p $@
	python3 -m src.mgp_tcn.mgp_tcn_fit -p -F $@ with $<
	@mv $@/1/* $@
	@rm -r $@/1
