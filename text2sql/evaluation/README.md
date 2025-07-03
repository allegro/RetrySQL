# Evaluation

## Benchmarks

Offline evaluation is implemented for the following benchmarks:
* `BIRD`: BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation) represents a pioneering,
cross-domain dataset that examines the impact of extensive database contents on text-to-SQL parsing ([source](https://bird-bench.github.io/))

## Metrics

Currently, we have two distinct groups of metrics implemented: text2sql and schema linking metrics.

### Text-to-SQL metrics

These metrics are directly related to the text2sql task. They basically consist of the implementation of the metrics
used in the aforementioned benchmarks.

* BIRD benchmark metrics - namely:
  * `Execution Accuracy (EX)` - EX is defined as the proportion of examples in the evaluation set for
which the executed results of both the predicted and ground-truth SQLs are identical, relative to the
overall number of SQLs;
  * `Valid Efficiency Score (VES)` - VES is a weighted version of EX, where the weight is a measure of
  the query efficiency. Among those correct SQL queries, it looks at how efficiently they are executed in terms of execution speed.
  Very long-executing queries are discarded and considered invalid.

### Schema linking metrics

These metrics are tailored to measure the quality of schema linking algorithms. Currently, the following metrics
have been implemented inspired by [Maamari et. al](https://arxiv.org/abs/2408.07702):

* `False Positive Rate (FPR)` - The proportion of irrelevant columns retrieved over the total number of retrieved columns;
* `Schema Linking Recall (SLR)` - The proportion of queries for which all required columns are retrieved.
* `Column Recall (CR)` - The fraction between actually retrieved columns over the total (ground truth) columns, averaged over all the test examples.

