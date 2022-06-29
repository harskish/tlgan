## AMOS download
The paper results are mainly generated using [AMOS data][amos]. The following instructions outline how to access and download the data:

1. Fill in the access request form (link on [this page][amos]) to request access to the Google Drive folder containing the AMOS data.
2. Install [rclone](https://rclone.org/install/), make sure it is in the PATH.
3. Install rclone-python: `pip install python-rclone`.
4. Create a [Google Application Client Id](https://rclone.org/drive/#making-your-own-client-id).
    - Take note of the `Client ID` and `Client secret` fields under `Credentials`.
5. Walk through the rclone Google Drive [setup guide](https://rclone.org/drive/).
    - Choose `No` when prompted to enter advanced config.
    - Choose `Yes` when asked if the target is a Shared Drive (Team Drive).
6. At the end of the config process, a block of the following format will be printed:
    ```
    --------------------
    [name]
    type = drive
    client_id = XXXXXX
    client_secret = YYYYYY
    scope = drive.readonly
    token = ZZZZZZ
    --------------------
    ```
    Set environment variables `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`, and `GDRIVE_TOKEN_JSON` to XXXXXX, YYYYYY, and ZZZZZZ respectively.

7. Try out the config by running `python preproc/AMOS/download.py --test`.
8. Specify which AMOS ids (cameras) to download in `preproc/AMOS/download.py` (bottom of the file).
    - `preproc/AMOS/good_cams.txt` contains some hand-picked stable cameras, although this list is by no means complete.
9. Start the download by running `python preproc/AMOS/download.py`. The files will be downloaded to `preproc/AMOS/AMOS_files/`.


## Dataset preprocessing
The input sequences (AMOS or otherwise) need to be processed by `preproc/process_sequence.py` in order to make sure the necessary metadata is created. The script can also be used to roughly align the input sequence, as well as crop, pad, and resize the images.

### Verify that image timestamps are working
1. Run script: `python preproc/process_sequence.py /path/to/frames`.
    - the target path will be recursively searched for folders or zip files containing images.
    - the timestamps of the images are parsed from the file names - the list of supported formats can be extended in `process_sequence.py:try_parse_filenames()`.
        - By default, only formats `20190521_150342` and `2019-05-21-15-03-42` are supported.
2. Scrub through the sequence and verify that the time and date in the UI appear correct (the time zone is assumed to be UTC).
3. The 'lock' button fixes the preview to the current time of day.

### Align images
In the UI:
1. Choose a beginning and end index with the `trim` sliders in the UI if the beginning or end of the sequence is not usable.
2. Scrub through the sequence, and press `spacebar` whenever the image alignment changes drastically. This creates a subsequence with internally more consistent alignment.
3. When all large aligments issues have been marked, go through the subsequences with `arrow left` and `arrow right`, and pick the same three points in the image with the `left mouse button`. Markers can be deleted with the `right mouse button`. The scroll wheel can be used to zoom and pan the view.
4. Export the alignment parameters by pressing `export warps`
    - this creates `path/to/frames/out/warps_manual.npy`

### Export sequence
In the UI:
1. Specify the `trim` sliders, see above.
2. Choose the newly exported `warps_manual.npy` in the `Warps` dropdown.
3. Click `fit window` to automatically crop in on the region of the frame that is visible in all aligned frames.
4. Specify additional padding or cropping.
5. Specify the desired output resolution.
6. Specify `skipped ranges` (e.g. `"10,25,100-200"`) to exclude frame ranges or individual frames from the export.
7. Click `export frames` to create `path/to/frames/out/NAME_WWWxHHH_XXXhz.zip`.
    - E.g. `muotathal_512x512_1200hz.zip` is a dataset with spatial resolution 512x512 of length 1200 days.

The exported dataset can now be used to train a TLGAN model.

[amos]: https://mvrl.cse.wustl.edu/datasets/amos/