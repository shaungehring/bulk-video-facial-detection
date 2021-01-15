import os
import cv2
from os import listdir
from os.path import isfile, join
import numpy
import click
import requests
import re
from pick import pick
import csv
import hashlib

VIDEO_STORAGE = "videos"
PROJECTS_FOLDER = "projects"
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

def project_selector():
    title = 'Please choose your project: '
    options = [name for name in os.listdir(PROJECTS_FOLDER) if os.path.isdir(os.path.join(PROJECTS_FOLDER, name)) ]
    option, index = pick(options, title)

    return option

def write_detection_log(project_folder, log):
    with open(join(project_folder, 'face_detection.csv'), 'w', newline='') as csvfile:
        cs_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cs_writer.writerow(['Face ID', 'Video ID', 'Timecode', 'Location'])

        for item in log:
            cs_writer.writerow(item)

@click.group()
def cli():
    pass

@cli.command()
def create_project():
    project_folder = click.prompt("What would you like to name the Project?")
    project_path = join(PROJECTS_FOLDER, project_folder)
    
    try:
        if not os.path.exists(PROJECTS_FOLDER):
            os.makedirs(PROJECTS_FOLDER)
        if not os.path.exists(project_path):
            os.makedirs(project_path)
            os.makedirs(f'{project_path}/videos')
            os.makedirs(f'{project_path}/faces')
            os.makedirs(f'{project_path}/people')
    except OSError:
        print('Error: Creating directory of data')

@cli.command()
def download_sample_videos():
    project_folder = project_selector()

    r = requests.get(VIDEO_PAGE)
    videos = re.findall(VIDEO_REGEX, r.text)

    with click.progressbar(videos, label="Video Download Progress", show_percent=True) as video_bar:
        for video in video_bar:
            url = f"https://www.tommycarstensen.com/terrorism/{video[1:]}"
            download_file(url, join(PROJECTS_FOLDER, project_folder, VIDEO_STORAGE))

@cli.command()
def detect_faces():
    project_folder = "Test"# project_selector()

    face_front_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
    face_profile_cascade = cv2.CascadeClassifier('classifiers/haarcascade_profileface.xml')

    video_files = listdir(join(PROJECTS_FOLDER, project_folder, VIDEO_STORAGE))

    detection_log = []

    with click.progressbar(video_files, label="Video Files Processing", show_percent=True) as video_files_bar:
        for f in video_files_bar:
            if isfile(join(PROJECTS_FOLDER, project_folder, VIDEO_STORAGE, f)):
                vidcap = cv2.VideoCapture(join(PROJECTS_FOLDER, project_folder, VIDEO_STORAGE, f))
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

                        if len(faces) > 0:
                            for (x, y, w, h) in faces:
                                try:
                                    # crop_face = gimg[y-10:y + h+10, x-10:x + w + 10]
                                    crop_face = image[y-10:y + h+10, x-10:x + w + 10]
                                    face_id = hashlib.md5(f'{f}_{count}_{fr}'.encode('utf-8')).hexdigest()

                                    cv2.imwrite(f"{join(PROJECTS_FOLDER, project_folder, 'faces')}/{face_id}.jpg", crop_face)
                                    detection_log.append([face_id, f, (float(time)/fps), f'{y-10}, {y + h+10}, {x-10}, {x + w + 10}'])
                                    fr += 1
                                except:
                                    pass

                    success, image = vidcap.read()

                    count += 1

                vidcap.release()
                cv2.destroyAllWindows()

        write_detection_log(join(PROJECTS_FOLDER, project_folder, 'faces'), detection_log)

@cli.command()
def find_people():
    project_folder = project_selector()
    faces_folder = join(PROJECTS_FOLDER, project_folder, 'faces')
    people_folder = join(PROJECTS_FOLDER, project_folder, 'people')




if __name__ == '__main__':
    cli()