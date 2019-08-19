import json
import os
import shutil
import pickle
import datetime
import numpy as np

class Data:
    '''A generic Data object to import and export data to and from JSON, respectively.

    Objects consist of two components: a meta dictionary and a data dictionary. Keys
    used in these dictionaries should be chosen consistently within data categories
    to allow reuse of data processing methods, such as plotting.

    '''
    def __init__(self, title):
        self.meta = {}
        self.meta['title'] = title
        self.data = {}

    def export(self, path, overwrite=False):
        '''Export the object's content to a .pickle file.

        Args:
            path (str): path to export to.
        '''
        # add export timestamp to meta
        self.meta['export_time_utc'] = str(datetime.datetime.utcnow())
        self.__sanitize(path)
        # open file
        if overwrite:
            if os.path.exists(self.json_path):
                os.remove(self.json_path)
            json_file = open(self.json_path, 'w')
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)
        else:
            if os.path.exists(path) or os.path.exists(self.json_path): # don't allow overwriting
                raise FileExistsError("File {} or directory {} already exist.".format(self.json_path, path))
            json_file = open(self.json_path, 'w')
            os.mkdir(path)
        # convert numpy arrays to lists
        for key in self.meta:
            if type(self.meta[key]) == np.ndarray:
                self.meta[key] = self.meta[key].tolist()
        # pickle data
        data_paths = {}
        for key in self.data.keys():
            p = '{}/{}.pickle'.format(path, key)
            data_paths[key] = p
            f = open(p, 'wb')
            pickle.dump(self.data[key], f)
            f.close()
        # export
        json_file.write(
            '{}\n\"META\":\n{},\n\n\"DATA_PATHS\":\n{}\n{}'.format(
                '{',
                json.dumps(self.meta, indent=2),
                json.dumps(data_paths, indent=2),
                '}'
            )
        )
        json_file.close()

    def load(path):
        out = Data(None)
        out.__sanitize(path)
        # open file
        if not (os.path.exists(path) and os.path.exists(out.json_path)):
            raise IOError("File {} or {} does not exists.".format(path, out.json_path))
        f = open(out.json_path, 'r')
        # import
        raw = json.load(f)
        f.close()
        out.meta = raw['META']
        # unpickle data
        for key in raw['DATA_PATHS'].keys():
            f = open(raw['DATA_PATHS'][key], 'rb')
            out.data[key] = pickle.load(f)
            f.close()
        return out

    def __repr__(self):
        return self.meta['title']

    def show(self):
        for key in self.meta.keys():
            print('{}:'.format(key))
            print('\t{}\n'.format(self.meta[key]))

    def __sanitize(self, path):
        if '.' in path:
            raise ValueError('Enter path without file extension!')
        self.json_path = '{}.json'.format(path)
