#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging

logger = logging.getLogger(__name__)

def generate_line_idx_file(in_file, out_file):
    tmp_out_file = out_file + '.tmp'
    with open(in_file, 'r') as f_in, open(tmp_out_file, 'w') as f_out:
        f_size = os.fstat(f_in.fileno()).st_size
        f_pos = 0
        while f_pos != f_size:
            f_out.write(str(f_pos) + '\n')
            f_in.readline()
            f_pos = f_in.tell()
    os.rename(tmp_out_file, out_file)


class TSVFile(object):
    def __init__(self, tsv_file, generate_line_idx=False):
        self.tsv_file = tsv_file
        self.line_idx = os.path.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._line_idx = None
        # The process always keeps the process which opens the file.
        # If the pid is not equal to the current pid, we will re-open the file.
        self.pid = None
        # Generate line idx if not exist
        if not os.path.isfile(self.line_idx) and generate_line_idx:
            generate_line_idx_file(self.tsv_file, self.line_idx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._line_idx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._line_idx[idx]
        except:
            logger.info('{}-{}'.format(self.tsv_file, idx))
            raise RuntimeError
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._line_idx is None:
            logger.info('Loading line idx from {}'.format(self.line_idx))
            with open(self.line_idx, 'r') as f:
                self._line_idx = [int(i.strip()) for i in f.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logger.info('Re-open {} because the process id changed!'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getid()
