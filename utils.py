from multiprocessing import managers
import torch
import RobotRL.tricks as tricks
import RobotRL.train as train

class RunningManager(managers.BaseManager):
    pass


def load_filter(dir, manager):
    original_filter = torch.load(dir)
    shape, _n, _M, _S, demean, discount, clip = original_filter.output()
    manager_filter = manager.ZFilter(shape)
    manager_filter.load(shape, _n, _M, _S, demean, discount, clip)
    return manager_filter


def save_filter(manager_filter, dir):
    shape, _n, _M, _S, demean, discount, clip = manager_filter.output()
    original_filter = tricks.ZFilter(shape)
    original_filter.load(shape, _n, _M, _S, demean, discount, clip)
    torch.save(original_filter, dir)


def load_buffer(dir, manager):
    original_buffer = torch.load(dir)
    memory, args = original_buffer.output()
    manager_buffer = manager.Buffer(args)
    manager_buffer.load(memory)
    return manager_buffer


def save_buffer(manager_buffer, dir):
    memory, args = manager_buffer.output()
    original_buffer = train.Buffer(args)
    original_buffer.load(memory)
    torch.save(original_buffer, dir)
