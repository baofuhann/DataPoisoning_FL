import torch
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client
from show.poisoning_data import showImg

def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch
    # 按照策略 选取本轮次进行本地训练的client
    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)
    random_workers[0] = poisoned_workers[0]
    random_workers[1] = poisoned_workers[1]
    print("恶意攻击者为",poisoned_workers)
    print("本轮训练者为:",random_workers)
    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        clients[client_idx].train(epoch)

    one = clients[random_workers[0]].test()
    two = clients[random_workers[1]].test()
    three = clients[random_workers[2]].test()
    four = clients[random_workers[3]].test()
    five = clients[random_workers[4]].test()

    local = [one, two, three, four, five]

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    new_nn_params = average_nn_parameters(parameters)

    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)



    return clients[0].test(), random_workers,local

def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients

def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    epoch_test_set_one = []
    epoch_test_set_two = []
    epoch_test_set_three = []
    epoch_test_set_four = []
    epoch_test_set_five = []

    worker_selection = []
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected,local = train_subset_of_clients(epoch, args, clients, poisoned_workers)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)
        epoch_test_set_one.append(local[0])
        epoch_test_set_two.append(local[1])
        epoch_test_set_three.append(local[2])
        epoch_test_set_four.append(local[3])
        epoch_test_set_five.append(local[4])

        localresult = [convert_results_to_csv(epoch_test_set_one),convert_results_to_csv(epoch_test_set_two),convert_results_to_csv(epoch_test_set_three),convert_results_to_csv(epoch_test_set_four),convert_results_to_csv(epoch_test_set_five)]
    return convert_results_to_csv(epoch_test_set_results), worker_selection,localresult


def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files,local_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)
    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    # 输出参数信息
    args.log()

    # batch_size = 10, data_num = 60000,则 data_loader的长度就是 60000/10 = 6000
    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    # Distribute batches equal volume IID
    # 对数据集进行划分，对50个worker进行均匀划分，worker_0的数据集就是distributed_train_dataset[0]
    # [[data,label],[data,label],[data,label],[data,label],.....,[data,label]]
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())

    # 生成投毒攻击的索引
    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    print("投毒攻击者",poisoned_workers)

    # 将 tensor 转成 array
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)
    #print("原始数据集",distributed_train_dataset[poisoned_workers[0]][1])
    # 根据投毒攻击者的索引，对数据进行投毒处理
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)
    #print("投毒数据集",distributed_train_dataset[poisoned_workers[0]][1])

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    clients = create_clients(args, train_data_loaders, test_data_loader)

    results, worker_selection,local = run_machine_learning(clients, args, poisoned_workers)
    save_results(results, results_files[0])
    # save_results(local[0], "3000_one.csv")
    # save_results(local[1], "3000_two.csv")
    # save_results(local[2],"3000_three.csv")
    # save_results(local[3], "3000_four.csv")
    # save_results(local[4], "3000_five.csv")
    save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)
