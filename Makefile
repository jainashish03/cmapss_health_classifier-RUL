.PHONY: install train compare all test app synthetic
install:
	pip install -r requirements.txt
train:                      ## train FD001 with the default model
	python -m cmapss_health.train --subset FD001
compare:                    ## cross-validate all candidate models on FD001, keep the best
	python -m cmapss_health.train --subset FD001 --compare
all:                        ## all four sub-datasets
	for s in FD001 FD002 FD003 FD004; do python -m cmapss_health.train --subset $$s --compare; done
synthetic:                  ## smoke test without the real data
	python -m cmapss_health.train --synthetic --subset FD002
test:
	python -m pytest -q tests
app:
	streamlit run app/streamlit_app.py
