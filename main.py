import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN

from generate_sample import generate_sample

import matplotlib.pyplot as plt

torch.manual_seed(0)

criterion = nn.CrossEntropyLoss()

f = None

def acc_plot(data, args, fname = None):
    plt.clf()
    if fname == None:
        fname = 'accuracies - %d samples - hyperparameters: lr=%f,num_layers=%d,num_mlp_layers=%d,hidden_dim=%d,batch_size=%d' % (len(data), args.lr, args.num_layers, args.num_mlp_layers, args.hidden_dim, args.batch_size)
    plt.title('%s - hyperparameters: lr=%f,num_layers=%d,num_mlp_layers=%d,hidden_dim=%d,batch_size=%d' % (fname, args.lr, args.num_layers, args.num_mlp_layers, args.hidden_dim, args.batch_size))
    for x, y, name in data:
        plt.plot(x, y, label = name)
    plt.legend(loc='lower right')
    plt.savefig('%s.png' % fname, dpi = 450)

def create_detail_plot_obj(train_labels, epochs):
    train_unique = np.unique(train_labels)

    res = []
    for i in range(len(train_unique)):
        res.append((range(epochs), [], 'train_%d_acc' % train_unique[i]))
    return res

def append_detail_plot_obj(obj, train_accs):
    combined = train_accs
    for i in range(len(combined)):
        obj[i][1].append(combined[i])

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch, f, train_acc_obj = None, valid_acc_obj = None):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    train_classes = [i.graph_class for i in train_graphs]
    total_classes = len(np.unique(train_classes))
    train_res = [0] * total_classes
    train_total_number = [0] * total_classes

    print('accuracy train: %f' % acc_train)
    f.write('train detailed acc: ')

    for i in range(len(train_graphs)):
        train_res[train_graphs[i].graph_class] += 1 if labels[i] == pred[i] else 0
        train_total_number[train_graphs[i].graph_class] += 1
    for i in range(total_classes):
        train_res[i] = float(train_res[i]) / train_total_number[i]
        print('  subclass %d accuracy: %f' % (i, train_res[i]))
        f.write('%f,' % train_res[i])

    f.write('\n')

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    
    test_classes = [i.graph_class for i in test_graphs]
    test_res = [0] * total_classes
    test_total_number = [0] * total_classes

    print('accuracy test: %f' % acc_test)
    f.write('test detailed acc: ')
    
    for i in range(len(test_graphs)):
        test_res[test_graphs[i].graph_class] += 1 if labels[i] == pred[i] else 0
        test_total_number[test_graphs[i].graph_class] += 1
    for i in range(total_classes):
        if test_total_number[i] != 0:
            test_res[i] = float(test_res[i]) / test_total_number[i]
            print('  subclass %d accuracy: %f' % (i, test_res[i]))
            f.write('%f,' % test_res[i])
    f.write('\n')

    if train_acc_obj != None:
        append_detail_plot_obj(train_acc_obj, train_res)
    if valid_acc_obj != None:
        append_detail_plot_obj(valid_acc_obj, test_res)

    return acc_train, acc_test

def main(debug = True):
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    parser.add_argument('--plot', type = str, default = 'plot_',
                                        help='output plot')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #loading data from data directory according to passing args
    data_dir = 'dataset/%s/%s' % (args.dataset, args.dataset)
    graphs, num_classes = load_data(data_dir + '.txt', args.degree_as_tag)
    train_graphs = graphs
    test_graphs, _ = load_data(data_dir + '_test.txt', args.degree_as_tag)
    valid_graphs, _ = load_data(data_dir + '_valid.txt', args.degree_as_tag)


    if debug:
        print('Debugging - load 1500 as training, 300 as validation, 1000 as testing')
        train_graphs = generate_sample(1500)
        valid_graphs = generate_sample(300)
        test_graphs = generate_sample(1000)

    #10-fold cross validation removed due to different file processing method
    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    #train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if not args.filename == '':
        f = open(args.filename, 'w')
        f.write('loss, train_acc, valid_acc\n')

    train_x = range(args.epochs+1)
    train_y = []
    train_name = 'train_acc'

    valid_x = range(args.epochs+1)
    valid_y = []
    valid_name = 'valid_acc'

    pre_acc_train, pre_acc_valid = test(args, model, device, train_graphs, valid_graphs, 0, f)

    train_y.append(pre_acc_train)
    valid_y.append(pre_acc_valid)

    print('Pre training: train: %f, test: %f' % (pre_acc_train, pre_acc_valid))

    train_acc_obj = create_detail_plot_obj([i.graph_class for i in train_graphs], args.epochs)
    valid_acc_obj = create_detail_plot_obj([i.graph_class for i in valid_graphs], args.epochs)

    #compute loss and accuracies of train, test, and validation for each epoch
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_valid = test(args, model, device, train_graphs, valid_graphs, epoch, f, train_acc_obj, valid_acc_obj)
        #acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)

        train_y.append(acc_train)
        valid_y.append(acc_valid)

        if not args.filename == "":
            f.write("%f,%f,%f" % (avg_loss, acc_train, acc_valid))
            f.write("\n")
            f.flush()
        print("")
        print(model.eps)
    if not args.filename == '':
        f.close()
    
    overall_acc_obj = [(train_x, train_y, train_name), (valid_x, valid_y, valid_name)]
    acc_plot(overall_acc_obj, args, fname = '%soverall accuracies_' % args.plot)
    acc_plot(train_acc_obj, args, fname = '%sspecific accuracies_train_' % args.plot)
    acc_plot(valid_acc_obj, args, fname = '%sspecific accuracies_valid_' % args.plot)

if __name__ == '__main__':
    main(debug = False)
