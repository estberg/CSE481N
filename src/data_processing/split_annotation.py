"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

"""
Split Annotation can be used after frames and information about 
each video have been generated (by extract_frames.py) to split
these into mutliple smaller sets. This  is used for getting 
samples with the first 100 class of labels, 500, 1000, etc. It
is not a representation of the size of the split.

Usage:

    python src/data_processing/split_annotation.py \
    -a {list of paths to text files containing annotations} \
    -k {subset of labels of splits}

Example:

    python src/data_processing/split_annotation.py \
    -a data/MS-ASL/frames/train.txt data/MS-ASL/frames/train.txt data/MS-ASL/frames/train.txt \
    -k 100 1000

The result will be in the same frames directory as the txt files 
containing the information of the samples to be used, and will be
txt files of the same format with the specified numbers in the splits.
"""

from os.path import exists, abspath
from argparse import ArgumentParser


class RawFramesSegmentedRecord(object):
    def __init__(self, row):
        self._data = row

        assert self.label >= 0

    @property
    def label(self):
        return int(self._data[1])

    @property
    def data(self):
        return self._data


def load_records(ann_file):
    return [RawFramesSegmentedRecord(x.strip().split(' ')) for x in open(ann_file)]


def filter_records(records, k):
    return [record for record in records if record.label < k]


def dump_records(records, out_file_path):
    with open(out_file_path, 'w') as out_stream:
        for record in records:
            out_stream.write('{}\n'.format(' '.join(record.data)))


def main():
    parser = ArgumentParser()
    parser.add_argument('--annot', '-a', nargs='+', type=str, required=True)
    parser.add_argument('--topk', '-k', nargs='+', type=int, required=True)
    args = parser.parse_args()

    records = dict()
    for annot_path in args.annot:
        assert exists(annot_path)

        records[annot_path] = load_records(annot_path)

    for k in args.topk:
        for annot_path in records:
            filtered_records = filter_records(records[annot_path], k)
            print(len(filtered_records))
            out_path = '{}{}.txt'.format(annot_path[:-len('.txt')], k)
            print(out_path)
            dump_records(filtered_records, out_path)


if __name__ == '__main__':
    main()