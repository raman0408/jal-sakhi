# Jal Sakhi C toolchain. On Windows, use build.ps1 instead of make.
CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Wno-unused-parameter
LDLIBS  := -lm

BINS    := build_dataset train_model
SAMPLES ?= 3200
SEED    ?= 11

.PHONY: all dataset train all-steps clean

all: $(BINS)

build_dataset: build_dataset.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

train_model: train_model.c scoring.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

dataset: build_dataset
	./build_dataset --synthetic $(SAMPLES) --seed $(SEED) --out dataset.csv

train: train_model
	./train_model --data dataset.csv --header model_params.h

all-steps: dataset train

clean:
	rm -f $(BINS) dataset.csv metrics.json simulation_report.json gateway_events.jsonl
