# 1. 각 peer에서 ratio대로 class imbalance가 지켜지면서 출력
# 2. 각 peer에서 출력되는 데이터의 class를 확인했는데, 최대한 같은 라벨끼리 출력되는것을 확인
# 3. 각각의 peer에서 같은 라벨에 대한 데이터를 출력할때, 데이터도 동일


# train_loader = New_DataLoader(dataset, sorted_dataset, dataset_name = 'cifar100', N_PEER =3, batch_size= 32, num_workers= 0, sampler = None, shuffle= True, ratio = 0.40)




#______________<인자 추가 목록>
# dataloader에 'N_PEER' 인자 추가
# 인자 'N_PEER': PEER의 개수(student의 수는 포함하지 X)

# dataloader에 'dataset_name' 인자 추가
# 'dataset_name'에 대한 설명
# 'cifar10' , 'cifar100' 등 데이터셋의 이름을 받는다. > class 개수를 얻는다. 

# 'sorted_dataset' : 라벨을 기준으로 데이터를 오름차순 정렬

# 'ratio' 인자 추가




import threading
import itertools
import warnings
from typing import Any, Callable, TypeVar, Generic, Sequence, List, Optional
from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union


import multiprocessing as python_multiprocessing
import torch
import torch.multiprocessing as multiprocessing
from torch._utils import ExceptionWrapper
from torch._six import queue, string_classes

from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import _utils

from new_sampler import Sampler, New_Sampler, New_Student_Sampler, New_BatchSampler, New_Student_BatchSampler
from new_fetch import New_MapDatasetFetcher



T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`, but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[List[T]], Any]


# This function used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate: _collate_fn_t = _utils.collate.default_collate

get_worker_info = _utils.worker.get_worker_info

class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, N_CLASS, N_PEER, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return New_MapDatasetFetcher(dataset, sorted_dataset, N_CLASS, N_PEER, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.
    Used as sampler for :class:`~torch.utils.data.IterableDataset`.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self):
        super(_InfiniteConstantSampler, self).__init__(None)

    def __iter__(self):
        while True:
            yield None


class New_DataLoader(Generic[T_co]):

    dataset: Dataset[T_co]
    dataset_name : str
    N_PEER : int
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable] 
    prefetch_factor: int
    _iterator : Optional['_BaseDataLoaderIter']
    __initialized = False
    ratio : float

    def __init__(self, dataset: Dataset[T_co], sorted_dataset, dataset_name = 'cifar100', N_PEER = 4, 
                 batch_size: Optional[int] = 1,
                 shuffle: bool = False, sampler: Optional[Sampler[int]] = None,
                 batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                 num_workers: int = 0, collate_fn: _collate_fn_t = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: _worker_init_fn_t = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False, 
                 ratio : float = 0.0):
        torch._C._log_api_usage_once("python.data_loader")  # type: ignore

        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if num_workers == 0 and prefetch_factor != 2:
            raise ValueError('prefetch_factor option could only be specified in multiprocessing.'
                             'let num_workers > 0 to enable multiprocessing.')
        assert prefetch_factor > 0

        if persistent_workers and num_workers == 0:
            raise ValueError('persistent_workers option needs num_workers > 0')

        self.dataset = dataset
        self.sorted_dataset = sorted_dataset
        self.dataset_name = dataset_name
    
        def get_n_classes(dataset):
            data_to_n_classes = {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000, 'cub200': 200, 'cars196': 196, 'dogs120': 120}
            assert dataset in list(data_to_n_classes.keys()), '{} is not included in data pool'.format(dataset)

            return data_to_n_classes[dataset]
    
        self.N_CLASS = int(get_n_classes(self.dataset_name))

        self.N_PEER = N_PEER
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.ratio = ratio




        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
       
            if shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))
            elif sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got samself.datasetpler={}".format(sampler))
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(batch_sampler))
        else:
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')
            
        #### 랜덤 인덱스 생성하기 > 나중에 데이터 섞는 역할    
        #### sampler에서 정의할 것임
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        # self.random_index = torch.randperm(50000, generator = generator).tolist()
        self.random_index = torch.randperm(50000, generator = generator).tolist()



       


        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    # Cannot statically verify that dataset is Sized
                    # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

                    # print(self.random_index[0:5])
                    # print('샘플러 전')

                    New_sampler = New_Sampler(self.dataset, self.ratio, self.random_index, self.N_CLASS)  # type: ignore
                    New_Student_sampler = New_Student_Sampler(self.dataset, self.random_index)  # New_Student_sampler에는 N_CLASS인자가 필요하지 않다. 

                else:
                    New_sampler = New_Sampler(self.dataset, self.ratio, self.random_index, self.N_CLASS )
                    New_Student_sampler = New_Student_Sampler(self.dataset, self.random_index)  # type: ignore


        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = New_BatchSampler(New_sampler, batch_size, drop_last)
            batch_Student_sampler = New_Student_BatchSampler(New_Student_sampler, batch_size, drop_last)


        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = New_sampler # new_sampler
        self.student_sampler = New_Student_sampler # new_sampler
        self.batch_sampler = batch_sampler
        self.student_batch_sampler = batch_Student_sampler

        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]

        self._iterator = None

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if not multiprocessing._supports_context:
                    raise ValueError('multiprocessing_context relies on Python >= 3.4, with '
                                     'support for different start methods')

                if isinstance(multiprocessing_context, string_classes):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            ('multiprocessing_context option '
                             'should specify a valid start method in {!r}, but got '
                             'multiprocessing_context={!r}').format(valid_start_methods, multiprocessing_context))
                    # error: Argument 1 to "get_context" has incompatible type "Union[str, bytes]"; expected "str"  [arg-type]
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)  # type: ignore

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise TypeError(('multiprocessing_context option should be a valid context '
                                     'object or a string specifying the start method, but got '
                                     'multiprocessing_context={}').format(multiprocessing_context))
            else:
                raise ValueError(('multiprocessing_context can only be used with '
                                  'multi-process loading (num_workers > 0), but got '
                                  'num_workers={}').format(self.num_workers))

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
                'batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(New_DataLoader, self).__setattr__(attr, val)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> '_BaseDataLoaderIter':
        # When using a single worker the returned iterator should be
        # created everytime to avoid reseting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):

        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.batch_sampler
        

    ########studnet index sampler추가
    @property
    def _student_index_sampler(self):

        if self._auto_collation:
            return self.student_batch_sampler
        else:
            return self.student_batch_sampler

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore
            if self.batch_size is not None:  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)


class _BaseDataLoaderIter(object):
    def __init__(self, loader: New_DataLoader) -> None:
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind

        # PEER의 개수/ CLASS의 개수를 받는다. 
        self._N_CLASS = loader.N_CLASS
        self._N_PEER = loader.N_PEER

        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        ########### student index sampler 추가
        self._student_index_sampler = loader._student_index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._pin_memory = loader.pin_memory and torch.cuda.is_available()
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        ########## student sampler iter 추가
        self._student_sampler_iter = iter(self._student_index_sampler)

        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self.random_index = loader.random_index

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._student_sampler_iter = iter(self._student_index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration
    
    def _student_next_index(self):
        return next(self._student_sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        if self._sampler_iter is None:
            self._reset()
        data = self._next_data()
        self._num_yielded += 1
        if self._dataset_kind == _DatasetKind.Iterable and \
                self._IterableDataset_len_called is not None and \
                self._num_yielded > self._IterableDataset_len_called:
            warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                        "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                              self._num_yielded)
            if self._num_workers > 0:
                warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                             "IterableDataset replica at each worker. Please see "
                             "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            warnings.warn(warn_msg)
    
        return data

    next = __next__  # Python 2 compatibility

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._N_CLASS, self._N_PEER, self._auto_collation, self._collate_fn, self._drop_last)
        

    def _next_data(self):
        index = self._next_index()
        student_index = self._student_next_index()  # may raise StopIteration
          # may raise StopIteration
        data = self._dataset_fetcher.fetch(index, student_index, self.random_index)  # may raise StopIteration

        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):

    

    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)

        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore
            # index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, self._collate_fn, self._drop_last,
                      self._base_seed + i, self._worker_init_fn, i, self._num_workers,
                      self._persistent_workers))
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      torch.cuda.current_device(),
                      self._pin_memory_thread_done_event))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        # Not that this indicates that a worker still has work to do *for this epoch*.
        # It does not mean that a worker is dead. In case of `_persistent_workers`, 
        # the worker will be reset to available in the next epoch.
        self._workers_status = [True for i in range(self._num_workers)]
        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(_utils.worker._ResumeIteration())
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                data = self._get_data()
                if isinstance(data, _utils.worker._ResumeIteration):
                    resume_iteration_cnt -= 1
        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `self._data_queue` once for a given timeout.
        # This can also be used as inner loop of fetching without timeout, with
        # the sender status as the loop condition.
        #
        # This raises a `RuntimeError` if any worker died expectedly. This error
        # can come from either the SIGCHLD handler in `_utils/signal_handling.py`
        # (only for non-Windows platforms), or the manual check below on errors
        # and timeouts.
        #
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
            if isinstance(e, queue.Empty):
                return (False, None)
            import tempfile
            import errno
            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                fs = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code") from None
            raise



    def _get_data(self):
       
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self._timeout))
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        while True:

            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _try_put_index(self):
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self._workers_status[worker_id] or (self._persistent_workers and shutdown)

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._workers_status[worker_id] = False

        assert self._workers_done_event.is_set() == shutdown

    def _shutdown_workers(self):
        # Called when shutting down this `_MultiProcessingDataLoaderIter`.
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, '_pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self._num_workers` in case we error before starting all
                    # workers.
                    # If we are using workers_status with persistent_workers
                    # we have to shut it down because the worker is paused
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False

    def __del__(self):
        self._shutdown_workers()

##########################

from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
import numpy as np
import random 
from tqdm import tqdm


if __name__ == "__main__":
    a = random.randint(0, 1000)
    random.seed(a)
    print('seed a:', a)


    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    dataset = CIFAR100(root='/disk/cifar100', train = True, download= True, transform = transform)

    # sorted_dataset 형태: 리스트 안에 튜플이 들어가있는 형태로, 인덱스를 부여했다.
    # sorted_dataset : class기준으로 오름차순 정렬
    # sorted_dataset의 i번째 원소: (i, (data, label))

    sorted_dataset = sorted(dataset, key = lambda x : x[1])  
    index_list = list(range(500))*100
    sorted_dataset = [(x, y) for x, y in zip(index_list, sorted_dataset)] 
    dataset = [(x, y) for x, y in zip(index_list, dataset)] 



    

    for i in range(5):


        train_loader = New_DataLoader(dataset, sorted_dataset, dataset_name = 'cifar100', N_PEER =3, batch_size= 32, num_workers= 0, sampler = None, shuffle= True, ratio = 0.40)

        N_CLASS = 100
        cnt = np.zeros((4, N_CLASS))

        for output in tqdm(train_loader):
            peer0_label = output[0][1]
            peer1_label = output[1][1]
            peer2_label = output[2][1]
            stu_label = output[3][1]



            # 각각의 peer에서 각 class를 몇개 씩 가지고 있는지 count하기
            for k in range(N_CLASS):
                cnt[0, k] += peer0_label.tolist().count(k)
                cnt[1, k] += peer1_label.tolist().count(k)
                cnt[2, k] += peer2_label.tolist().count(k)
                cnt[3, k] += stu_label.tolist().count(k)

        print('peer0의 class별 출력개수')
        print(cnt[0,:])
        print('-'*20)
        print('peer1의 class별 출력개수')
        print(cnt[1,:])
        print('-'*20)
        print('peer2의 class별 출력개수')
        print(cnt[2,:])
        print('-'*20)
        print('stu의 class별 출력개수')
        print(cnt[3,:])
        print('-'*20)
        print('출력종료')



   
