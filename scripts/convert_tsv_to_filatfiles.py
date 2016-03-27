# Usage: cat data.tsv | python ./scripts/convert_tsv_to_flatfiles.py flatfiles.config output_dir
import csv
import json
import os
import sys

def CreateWriters(output_flatfiles_dir, header, raw_float_columns, string_columns):
    flatfiles = [os.path.join(output_flatfiles_dir, column.replace(' ', '_').replace('/', '_'))
                 for column in header]
    writers = [open(flatfile, 'w') for flatfile in flatfiles]
    for i, column in enumerate(header):
        if column in raw_float_columns:
            print >>writers[i], '# dtype=raw_floats'
        elif string_columns and column in string_columns:
            print >>writers[i], '# dtype=strings'
        else:
            print >>writers[i], '# dtype=binned_floats'
    print '\n'.join(['Writing to %s' % flatfile for flatfile in flatfiles])
    return writers


def ConvertTsvToFlatfiles(in_file, output_flatfiles_dir, delimiter,
                          raw_float_columns, string_columns):
    reader = csv.reader(in_file, delimiter=str(delimiter))
    writers = CreateWriters(output_flatfiles_dir, reader.next(), raw_float_columns, string_columns)

    for row in reader:
        for i, value in enumerate(row):
            print >>writers[i], value


if __name__ == '__main__':
    config_file = sys.argv[1]
    output_flatfiles_dir = sys.argv[2]
    in_file = sys.stdin
    config = json.loads(open(config_file).read())

    if not os.path.exists(output_flatfiles_dir):
        print "Creating %s " % output_flatfiles_dir
        os.mkdir(output_flatfiles_dir)
    ConvertTsvToFlatfiles(in_file, output_flatfiles_dir,
                          delimiter=config.get('delimiter', '\t'),
                          raw_float_columns=config.get('raw_float_columns', []),
                          string_columns=config.get('string_columns', []))
