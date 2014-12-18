import numpy as np
import os
import sys
import fcntl
import copy
from string import Template
import mlpython.datasets.store as dataset_store
import mlpython.mlproblems.generic as mlpb
from temporal_nnet import TemporalNeuralNetwork
import dataset

sys.argv.pop(0)
# Check if every option(s) from parent's script are here.
if 8 != len(sys.argv):
    print "Usage: python run_temporal_nnet.py lrFirstPhase lrSecondPhase dc sizes(1x4) seed batchsize lookahead_steps lookahead_delay"
    print ""
    print "Ex.: python run_temporal_nnet.py 0.001 0.005 0 [80,40,20,10] 1234 6 10 7"
    print "received = " 
    print sys.argv
    sys.exit()

# Set the constructor
#sys.argv[1] = sys.argv[0] #FIXME : Remove after tests.
str_ParamOption = "lrFirstPhase=" + sys.argv[0] + ", " + "lrSecondPhase="+ sys.argv[1] + ", " + "dc=" + sys.argv[2] + ", " + "sizes=" + sys.argv[3] + ", " + "seed=" + sys.argv[4]

str_ParamOptionValue = sys.argv[0] + "\t" + sys.argv[1] + "\t" + sys.argv[2] + "\t" + sys.argv[3] + "\t" + sys.argv[4] + "\t" +sys.argv[5] + "\t" + sys.argv[6] + "\t" + sys.argv[7] + "\t"
look_ahead_delay = int(sys.argv[7])
look_ahead = int(sys.argv[6])
batchSize = int(sys.argv[5])

try:
    objectString = 'myObject = TemporalNeuralNetwork(n_epochs=1,' + str_ParamOption + ')'
    exec objectString
    # code = compile(objectString, '<string>', 'exec')
    #exec code
except Exception as inst:
    print "Error while instantiating NeuralNetwork (required hyper-parameters are probably missing)"
    print inst

print "Loading dataset..."
#trainset, validset, testset = dataset_store.get_classification_problem('ocr_letters')
trainset, validset, testset = dataset.get_classification_problem('/home/haze/Documents/repositories/images/', batchSize)


print "Training..."
# Early stopping code
best_val_error = np.inf
best_it = 0
str_header = 'best_it\t'

n_incr_error = 0
for stage in range(1, 500 + 1, 1):
    if not n_incr_error < look_ahead:
        break
    myObject.n_epochs = stage
    myObject.train(trainset)
    if stage >= look_ahead_delay:
        n_incr_error += 1
    outputs, costs = myObject.test(trainset)
    errors = np.mean(costs, axis=0)
    print 'Epoch', stage, '|',
    print 'Training errors: classif=' + '%.3f' % errors[0] + ',', 'NLL=' + '%.3f' % errors[1] + ' |',
    outputs, costs = myObject.test(validset)
    errors = np.mean(costs, axis=0)
    print 'Validation errors: classif=' + '%.3f' % errors[0] + ',', 'NLL=' + '%.3f' % errors[1]
    error = errors[0]
    if error < best_val_error:
        best_val_error = error
        best_it = stage
        n_incr_error = 0
        best_model = copy.deepcopy(myObject)

outputs_tr, costs_tr = best_model.test(trainset)
columnCount = len(costs_tr.__iter__().next())
outputs_v, costs_v = best_model.test(validset)
outputs_t, costs_t = best_model.test(testset)

# Preparing result line
str_modelinfo = str(best_it) + '\t'
train = ""
valid = ""
test = ""
# Get average of each costs
for index in range(columnCount):
    train = str(np.mean(costs_tr, axis=0)[index])
    valid = str(np.mean(costs_v, axis=0)[index])
    test = str(np.mean(costs_t, axis=0)[index])
    str_header += 'train' + str(index + 1) + '\tvalid' + str(index + 1) + '\ttest' + str(index + 1)
    str_modelinfo += train + '\t' + valid + '\t' + test
    if ((index + 1) < columnCount):  # If not the last
        str_header += '\t'
        str_modelinfo += '\t'
str_header += '\n'
result_file = '/home/haze/Dropbox/Ulaval/Maitrise - Session 1/Neural Networks/Travaux/results/results_temporal_nnet_ocr_letters.txt'

# Preparing result file
header_line = ""
header_line += 'lrFirstPhase\tlrSecondPhase\tdc\tsizes\tseed\tbatchsize\tlook_ahead_steps\tlook_ahead_delay'
header_line += str_header
if not os.path.exists(result_file):
    f = open(result_file, 'w')
    f.write(header_line)
    f.close()

# Look if there is optional values to display
if str_ParamOptionValue == "":
    model_info = [str_modelinfo]
else:
    model_info = [str_ParamOptionValue, str_modelinfo]

line = '\t'.join(model_info) + '\n'
f = open(result_file, "a")
fcntl.flock(f.fileno(), fcntl.LOCK_EX)
f.write(line)
f.close()  # unlocks the file

