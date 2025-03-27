from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import replace_6_with_0
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
import time

if __name__ == '__main__':
    START_EXP_IDX = 3000 # 实验编号（用于保存日志或结果）起始编号
    NUM_EXP = 1 # 实验次数，这里只跑一轮
    NUM_POISONED_WORKERS = 3 # 恶意节点数量
    REPLACEMENT_METHOD = replace_4_with_6 # 攻击方法：将标签 4 改为标签 6
    KWARGS = {
        "NUM_WORKERS_PER_ROUND" : 5 # 每轮选择的工作节点数量
    }
    start = time.time()

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):

        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))