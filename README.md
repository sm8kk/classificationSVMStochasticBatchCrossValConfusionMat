# classificationSVMStochasticBatchCrossValConfusionMat
Usage: time python classify-for-svm-rbf-space.py schuffled-numeric-ps-to-fdt-one-iperf-load-btlnck-cpu-transfer-expts-all.csv c-gamma-rbf-diag.csv > rbf-C-gamma-runs.txt

"classify-for-svm-rbf-space.py" runs an SVM classifier for a "rbf" kernel (Radial basis function) over a range of values of C and gamma. The cross validation used is a stochastic batch cross validation technique. We use a confusion matrix score to test the impact of C and gamma for measuring classification accuracy.
