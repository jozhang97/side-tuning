import warnings

def cycle_dl(dl):
    while True:
        for element in dl:
            yield element
    # cycle('ABCD') --> A B C D A B C D A B C D ...
    # version of cycle where the dataloader repeats itself
    # this is how itertools.cycle works  (huge saved cache)
    #   saved = []
    #   for element in iterable:
    #       yield element
    #       saved.append(element)
    #   while saved:
    #       for element in saved:
    #           yield element

class CyclingDataLoader(object):

    def __init__(self, dls, epoch_length_per_dl=None, start_dl=0, epochs_until_cycle=0, zip_idx=True):
        """
        :param dls: list of dataloaders, one for each task
        :param epoch_length_per_dl: number of items to cycle thru dataset
        :param start_dl:
        :param epochs_until_cycle: num epochs to train on the first task exclusively before training on others
        :param zip_idx: when calling __next__, return the task_idx as well
        """
        self.epochlength = epoch_length_per_dl
        self.dls = dls
        self.curr_iter_idx = start_dl - 1
        self.start_dl = start_dl
        self.epochs_until_cycle = epochs_until_cycle + 1
        self.zip_idx = zip_idx

    def __iter__(self):
        self.curr_iter_idx = (self.curr_iter_idx + 1) % len(self.dls)
        if self.epochs_until_cycle > 0:
            self.curr_iter_idx = self.start_dl
            self.epochs_until_cycle -= 1
        self.curr_iter = cycle_dl(self.dls[self.curr_iter_idx])
        self.count = 0
        return self

    def __next__(self):
        # this is one batch, so need to stop after n batches, not self.epochlength
        if self.count == len(self):
            raise StopIteration
        self.count += 1
        if self.zip_idx:
            return self.curr_iter_idx , next(self.curr_iter)
        return next(self.curr_iter)

    @property
    def batch_size(self):
        return self.dls[self.curr_iter_idx].batch_size

    def __len__(self):
        if self.epochlength is None:
            return len(self.dls[self.curr_iter_idx])
        return self.epochlength // self.dls[self.curr_iter_idx].batch_size + 1
        # return self.epochlength   this is wrong - it is the dataset size but dataloader len should be number of batches

    def get_last_dl(self):
        # need this if post_training (e.g. EWC) requires a pass over the data - do not want to move to next task
        return self.curr_iter_idx, self.dls[self.curr_iter_idx]


class ErrorPassingCyclingDataLoader(CyclingDataLoader):
    def __next__(self):
        try:
            return super().__next__()
        except Exception as e:
            if isinstance(e, StopIteration):
                raise e
            else:
                warnings.warn('problem with this datapoint, resampling')
                print(e)
                return self.__next__()

class ConcatenatedDataLoader(object):
    def __init__(self, dls, zip_idx=True):
        self.dls = dls
        self.curr_iter_idx = 0
        self.zip_idx = zip_idx

    def __iter__(self):
        self.curr_iter_idx = 0
        self.curr_iter = iter(self.dls[self.curr_iter_idx])
        return self

    def __next__(self):
        try:
            if self.zip_idx:
                return self.curr_iter_idx , next(self.curr_iter)
            return next(self.curr_iter)
        except StopIteration:
            self.curr_iter_idx += 1
            if self.curr_iter_idx >= len(self.dls):
                self.curr_iter_idx = 0  # reset
                raise StopIteration
            self.curr_iter = iter(self.dls[self.curr_iter_idx])
            return self.__next__()

    def __len__(self):
        return sum([len(d) for d in self.dls])

class ErrorPassingConcatenatedDataLoader(ConcatenatedDataLoader):
    def __next__(self):
        try:
            return super().__next__()
        except Exception as e:
            if isinstance(e, StopIteration):
                raise e
            else:
                warnings.warn('problem with this datapoint, resampling')
                print(e)
                return self.__next__()

class KthDataLoader(object):

    def __init__(self, dls, k=0, epochlength=None):
        self.dls = dls
        self.dl = dls[k]
        self.k = k
        self.epochlength = epochlength

    def __iter__(self):
        self.count = 0
        if self.epochlength:
            self.curr_iter = cycle_dl(self.dl)
        else:
            self.curr_iter = iter(self.dl)
        return self

    def __next__(self):
        if self.epochlength and self.count == self.epochlength:
            raise StopIteration
        self.count += 1
        return next(self.curr_iter)

    def __len__(self):
        return len(self.dl)
