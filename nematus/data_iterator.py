import numpy

import gzip

import shuffle
from util import load_dict

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, sources, target,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=[-1],
                 n_words_target=-1,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 maxibatch_size=20):
        if shuffle_each_epoch:
            shuffle.main(sources + [target])
            self.sources = [fopen(source+'.shuf', 'r') for source in sources]
            self.target = fopen(target+'.shuf', 'r')
        else:
            self.sources = [fopen(source, 'r') for source in sources]
            self.target = fopen(target, 'r')
        self.source_dicts = []
        for factor_dicts in source_dicts:
            self.source_dicts.append([load_dict(source_dict) for source_dict in factor_dicts])
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        for i, n_words in enumerate(self.n_words_source):
            if n_words > 0:
                for d in self.source_dicts[i]:
                    for key, idx in d.items():
                        if idx >= n_words:
                            del d[key]

        if self.n_words_target > 0:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffers = [list() for _ in range(len(self.sources))]
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            shuffle.main([source.name.replace('.shuf','') for source in self.sources] + [self.target.name.replace('.shuf','')])
            self.sources = [fopen(source.name, 'r') for source in self.sources]
            self.target = fopen(self.target.name, 'r')
        else:
            for source in self.sources:
                source.seek(0)
            self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        sources = []
        target = []

        for source_buffer in self.source_buffers:
            assert len(source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        # fill buffer, if it's empty
        if len(self.source_buffers[0]) == 0:
            for k_ in xrange(self.k):
                doBreak = False
                for i, source in enumerate(self.sources):
                    ss = source.readline()
                    if ss == "":
                        doBreak = True
                        break
                    self.source_buffers[i].append(ss.strip().split())
                if doBreak:
                    break
                tt = self.target.readline()
                if tt == "":
                    break

                self.target_buffer.append(tt.strip().split())
            
            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()
                _sbufs = []
                for source_buffer in self.source_buffers:
                    _sbufs.append([source_buffer[i] for i in tidx])
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffers = _sbufs
                self.target_buffer = _tbuf

            else:
                for source_buffer in self.source_buffers:
                    source_buffer.reverse()
                self.target_buffer.reverse()


        if any(len(s) == 0 for s in self.source_buffers) or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            # actual work here
            while True:
                sss = []
                try:
                    for k, source_buffer in enumerate(self.source_buffers):
                    # read from source file and map to word index
                        ss = source_buffer.pop()
                        tmp = []
                        for w in ss:
                            w = [self.source_dicts[k][i][f] if f in self.source_dicts[k][i] else 1 for (i,f) in enumerate(w.split('|'))]
                            tmp.append(w)
                        ss = tmp
                        sss.append(ss)
                except IndexError:
                    break
                
                # read from target file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if all(len(ss) > self.maxlen for ss in sss) and len(tt) > self.maxlen:
                    continue

                sources.append(sss)
                target.append(tt)

                if len(sources) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(sources) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return sources, target
