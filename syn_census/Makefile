# Input data 
AL_DATA = ~/Desktop/census_data/output/AL/21877344_synthetic.csv

# Output files
AL_RESULTS = AL_results.tex

AL_EVALUATION = AL_evaluation.tex

# Processing script
PROCESSOR = get_synthetic_stats.py 
EVALUATOR = sampling_evaluation.py

all: $(AL_RESULTS) $(AL_EVALUATION)

$(AL_RESULTS): $(PROCESSOR) $(AL_DATA)
	python3 $(PROCESSOR) $(AL_DATA) "AL" > $(AL_RESULTS)

$(AL_EVALUATION): $(EVALUATOR)
	python3 $(EVALUATOR) --from_params AL_params.json --task_name "AL_evaluation" > $(AL_EVALUATION)

clean:
		rm -f $(AL_RESULTS) $(AL_EVALUATION)

.PHONY: clean
