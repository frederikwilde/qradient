import json
import os
import shutil
import pickle
import datetime
import numpy as np
from logging import debug, info, warn

import warnings
warnings.warn(
    'Data storage is only supported up to version 1.0. Use HDF5 to store data.',
    DeprecationWarning,
    stacklevel=2
)

class Data:
    '''A generic Data object to import and export data to and from JSON, respectively.

    Objects consist of two components: a meta dictionary and a data dictionary. Keys
    used in these dictionaries should be chosen consistently within data categories
    to allow reuse of data processing methods, such as plotting.
    '''
    def __init__(self, path):
        self.meta = {}
        self.data = {}
        self.__path = sanitize(path)
        self.__working_directory = os.getcwd()
        debug('New data object. Export will dump content to {}/{}'.format(
            self.__working_directory,
            self.__path
        ))

    def export(self):
        '''Export the object's content to a .pickle file.'''
        # add export timestamp to meta
        self.meta['export_time_utc'] = str(datetime.datetime.utcnow())
        # convert numpy arrays to lists
        for key in self.meta:
            if type(self.meta[key]) == np.ndarray:
                self.meta[key] = self.meta[key].tolist()
        create_missing_directories(self.__path)
        # check if data already exists
        export_path = secure_export_path(self.__path)
        # make directory for pickle data
        os.mkdir(export_path)
        # pickle data
        data_paths = {}
        for key in self.data.keys():
            p = '{}/{}.pickle'.format(export_path, key)
            data_paths[key] = p
            f = open(p, 'wb')
            pickle.dump(self.data[key], f)
            f.close()
        # export meta to JSON file
        json_file = open(export_path+'.json', 'w')
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
        out = Data(path)
        json_path = out.__path + '.json'
        # open file
        if not (os.path.exists(out.__path) and os.path.exists(json_path)):
            raise IOError("Directory {} or file {} does not exists in {}.".format(
                out.__path,
                json_path,
                out.__working_directory
            ))
        f = open(json_path, 'r')
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

    def show(self):
        for key in self.meta.keys():
            print('{}:'.format(key))
            print('\t{}\n'.format(self.meta[key]))

def secure_export_path(path):
    data_exists = os.path.exists(path + '.json') or os.path.exists(path)
    if data_exists:
        export_path = path + '-1'
        counter = 2
        while True:
            if os.path.exists(export_path + '.json') or os.path.exists(export_path):
                export_path = export_path[:-1] + str(counter)
                counter += 1
            else:
                break
    else:
        export_path = path
    if data_exists:
        info(''.join([
            'There is already data at {}.'.format(path),
            'Export will be dumped to {}.'.format(export_path)
        ]))
    else:
        debug('Exporting data to {}.'.format(export_path))
    return export_path

def create_missing_directories(path):
    path_components = path.split('/')
    if len(path_components) > 1: # if storage into a subdirectory is wanted
        for i in range(1, len(path_components)):
            dir = '/'.join(path_components[:i])
            if not os.path.exists(dir):
                os.mkdir(dir)
                info('Directory {} did not exist, it was created.'.format(dir))

def sanitize(path):
    if type(path) != str:
        warn(''.join([
            'Path has type {}. Should be string. '.format(type(path)),
            'Path has been set to: {}/output'.format(os.getcwd())
        ]))
        return('output')
    if path == '':
        warn(''.join([
            'Path was an empty string. ',
            'Path has been set to: {}/output'.format(os.getcwd())
        ]))
        return('output')
    path_components = path.split('/')
    name = path_components[-1]
    name_components = name.split('.')
    for comp in name_components:
        if len(comp) != 0:
            path_components[-1] = comp
            break
    new_path = '/'.join(path_components).replace(' ', '_')
    if new_path[-1] == '/':
        new_path = new_path[:-1]
    if new_path != path:
        info('Path has been sanitized. It is now: {}'.format(new_path))
    return new_path
