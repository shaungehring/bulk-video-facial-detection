import os
import cv2
from os import listdir
from os.path import isfile, join
import numpy
import click
import requests
import re

VIDEO_STORAGE = "videos/"
VIDEO_PAGE = "https://www.tommycarstensen.com/terrorism/index.html"
VIDEO_REGEX = r"\".*\.mp4"


def download_file(url, dl_location):
    local_filepath = join(dl_location, url.split('/')[-1])

    if not os.path.isfile(local_filepath):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filepath
    else:
        return None

@click.group()
def cli():
    pass

@cli.command()
def download_videos():
    click.echo('Grabbing Videos')

    try:
        if not os.path.exists(VIDEO_STORAGE):
            os.makedirs(VIDEO_STORAGE)
    except OSError:
        print(f'Error: Creating directory of {VIDEO_STORAGE}')

    r = requests.get(VIDEO_PAGE)
    videos = re.findall(VIDEO_REGEX, r.text)

    with click.progressbar(videos, label="Video Download Progress", show_percent=True) as video_bar:
        for video in video_bar:
            url = f"https://www.tommycarstensen.com/terrorism/{video[1:]}"
            download_file(url, VIDEO_STORAGE)

@cli.command()
def detect_faces():
    click.echo('Detecting Faces')
    face_front_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
    face_profile_cascade = cv2.CascadeClassifier('classifiers/haarcascade_profileface.xml')

    try:
        if not os.path.exists('all_frames'):
            os.makedirs('all_frames')
    except OSError:
        print('Error: Creating directory of data')

    video_files = listdir(VIDEO_STORAGE)

    with click.progressbar(video_files, label="Video Files Processing", show_percent=True) as video_files_bar:
        for f in video_files_bar:
            if isfile(join(VIDEO_STORAGE, f)):
                vidcap = cv2.VideoCapture(join(VIDEO_STORAGE, f))

                count = 0

                success, image = vidcap.read()
                while success:
                    time = round(vidcap.get(cv2.CAP_PROP_POS_MSEC), 0)
                    fps = round(vidcap.get(cv2.CAP_PROP_FPS), 0)-1

                    if count % fps < 1:
                        gimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        face_fronts = face_front_cascade.detectMultiScale(gimg, scaleFactor=1.1, minNeighbors=5)
                        face_profiles = face_profile_cascade.detectMultiScale(gimg, scaleFactor=1.1, minNeighbors=5)

                        try:
                            faces = numpy.concatenate((face_fronts, face_profiles))
                        except:
                            faces = face_fronts

                        fr = 0

                        for (x, y, w, h) in faces:
                            try:
                                # crop_face = gimg[y-10:y + h+10, x-10:x + w + 10]
                                crop_face = image[y-10:y + h+10, x-10:x + w + 10]
                                cv2.imwrite(f"./all_frames/{f}_frame{count}Person{fr}.jpg", crop_face)
                                fr += 1
                            except:
                                pass

                    success, image = vidcap.read()

                    count += 1

                vidcap.release()
                cv2.destroyAllWindows()

if __name__ == '__main__':
    cli()