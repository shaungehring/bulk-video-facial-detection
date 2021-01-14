## Bulk Face Detection

This script will look at a directory of videos and try to detect all of the faces in that video. It will save each face into a 'all_frames' directory. 

#

### Usage

Install the requirements

```
pip install -r requirements.txt
```

(Optional) Download Test videos
```
python detector.py download-videos
```
This will download all of the Parlor archived videos from January 6, 2021 at the Capitol. You can also just create a `videos` folder and fill it with your own.

Detect Faces
```
python detector.py detect-faces
```

This will process all videos in the folder and look for frontal and profile faces and put them in the `all_frames` folder it creates.

### Upcoming

I am looking to add facial grouping, or comparing the pictures to find the same faces throughout.