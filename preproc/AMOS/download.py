# Copyright (c) 2022 Erik Härkönen, Aalto University
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

import rclone # pip install python-rclone
import os
import sys
from math import floor, log
from pathlib import Path
from datetime import datetime
from tqdm import trange

# By hand:
# rclone ls --drive-shared-with-me amos:AMOS_data/2002
#   => 138480 04/1804/00021804/2002.02.zip
# rclone copy --drive-shared-with-me amos:AMOS_data/2002/04/1804/00021804/2002.02.zip D:\datasets\timelapse\AMOS\

class FSItem:
    def __init__(self, desc, path_prefix=None) -> None:
        parts = desc.strip().split()
        self.size_bytes = int(parts[0])
        self.size = self.format_size(self.size_bytes)
        mod_time = parts[2].split('.')[0] # remove ms
        self.modified = datetime.strptime(f'{parts[1]} {mod_time}', r'%Y-%m-%d %H:%M:%S')
        self.path = f'{path_prefix}/{parts[-1]}' if path_prefix else parts[-1]
        self.name = self.path.split('/')[-1]
        self.is_dir = (self.size_bytes == -1)
    
    def format_size(self, size_bytes):
        if size_bytes < 1:
            return '0B'
        
        prefix = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']
        idx = floor(log(size_bytes, 1000))
        return f'{size_bytes//(1000**idx)}{prefix[idx]}'

    def __repr__(self) -> str:
        return f'{self.path} {self.size}'

# Mirror remote file system locally, return local
# copies if available to avoid unnecessary transfers
class GDriveMirror:
    def __init__(self, local_root, client_id, client_secret, token) -> None:
        # Created with 'rclone config'; see https://rclone.org/drive/
        # Client managed at https://console.cloud.google.com/apis/
        cfg = """
            [cfg]
            type = drive
            client_id = %s
            client_secret = %s
            scope = drive.readonly
            token = %s
        """ % (client_id, client_secret, token)

        assert all(k in token for k in ['access_token', 'token_type', 'refresh_token', 'expiry']), \
            'Token must be a json-formatted string of the format {"access_token":"XXX","token_type":"Bearer","refresh_token":"YYY","expiry":"ZZZ"}'
        
        self.cfg = cfg.strip()
        self.root = Path(local_root)
        os.makedirs(self.root, exist_ok=True)

    # Get file from from local mirror, or transfer from remote
    def get(self, item):
        assert isinstance(item, FSItem)
        assert not item.is_dir, 'Avoid directory transfers'

        cached = self.to_local_path(item.path)
        if not cached.is_file():
            # For directory transfers: "--transfers=32", "--checkers=16", "--drive-chunk-size=16384k", "--drive-upload-cutoff=16384k"
            print(f'Transferring file {item.path} ({item.size})')
            os.makedirs(cached.parent, exist_ok=True)
            result = rclone.with_config(self.cfg).run_cmd(command="copy", extra_args=["--drive-shared-with-me", f"cfg:AMOS_data/{item.path}", str(cached.parent)])
            if result['error'] != '':
                print('An error occured:\n', result['error'])
        
        return cached
    
    # List directories, non-recursive
    def lsd(self, path, print_err=True):
        result = rclone.with_config(self.cfg).run_cmd(command="lsd", extra_args=["--drive-shared-with-me", f"cfg:AMOS_data/{path}"], print_err=print_err)
        err = result['error']
        if err:
            if 'Unauthorized' in err:
                raise RuntimeError('RClone: unauthorized')
            elif 'directory not found' in err:
                return []
            else:
                raise RuntimeError(f'RClone: unknown error: {err}')
        else:
            items = [FSItem(l, path) for l in result.get('out').splitlines()]
            return [i for i in items if i.is_dir]

    # List files, recursive
    def lsf(self, path):
        result = rclone.with_config(self.cfg).run_cmd(command="lsl", extra_args=["--drive-shared-with-me", f"cfg:AMOS_data/{path}"])
        return [FSItem(l, path) for l in result.get('out').splitlines()]

    # Structure: year/last2/last4/whole_id/zips
    def list_cams(self, y):
        flatten = lambda t: [item for sublist in t for item in sublist]
        ids_l2 = self.lsd(y)
        ids_l4 = flatten([self.lsd(l2.path) for l2 in ids_l2])
        ids = flatten([self.lsd(l4.path) for l4 in ids_l4])
        return [c.name for c in ids]

    def id_to_path(self, cam_id):
        cam_id = '{:08d}'.format(int(cam_id))
        return f'{cam_id[-2:]}/{cam_id[-4:]}/{cam_id}'

    # Convert remote path (year/id1/id2/id3/year.month.zip)
    # to local path (id3/year.month.zip)
    def to_local_path(self, remote_path):
        year, id1, id2, id3, fname = remote_path.split('/')
        return self.root / id3 / fname

    # List years from which webcam has data
    def list_years(self, cam_id):
        cam_id = '{:08d}'.format(int(cam_id))
        matches = []
        
        for y in range(2002, 2018):
            dirs_full = self.lsd(f'{y}/{cam_id[-2:]}/{cam_id[-4:]}', print_err=False)
            if dirs_full and cam_id in [d.name for d in dirs_full]:
                matches.append(y)

        return matches

    def list_files(self, y, cam_id):
        return self.lsf(f'{y}/{self.id_to_path(cam_id)}')

# Make sure env is set up correctly
assert 'GDRIVE_CLIENT_ID' in os.environ, 'GDRIVE_CLIENT_ID not in PATH'
assert 'GDRIVE_CLIENT_SECRET' in os.environ, 'GDRIVE_CLIENT_SECRET not in PATH'
assert 'GDRIVE_TOKEN_JSON' in os.environ, 'GDRIVE_TOKEN_JSON not in PATH'

local_root = Path(__file__).parent / 'AMOS_files'
fs = GDriveMirror(str(local_root), os.environ['GDRIVE_CLIENT_ID'],
    os.environ['GDRIVE_CLIENT_SECRET'], os.environ['GDRIVE_TOKEN_JSON'])

if '--test' in sys.argv:
    res = fs.lsd(2002)
    assert res, 'ERROR: Could not get listing for year 2002'
    print('Success!')
    sys.exit(0)

# Download target zips
targets = [
    19189, # Barn
     9483, # Frankfurt
     8687, # Küssnacht
    10180, # Muotathal
     9780, # Normandy
    19188, # Teton
    17183, # Two Medicine
     7371, # Valley
]

for target in targets:
    years = fs.list_years(target)[::-1] # start with newest data
    if len(years) == 0:
        print(f'No data found for {target}')
    for i in trange(len(years)):
        zips = fs.list_files(years[i], target)
        for z in zips:
            fs.get(z)

print('Done')
