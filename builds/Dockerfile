# Make base state
FROM python:3.10 as base_python3.10

# install dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install zip vim default-mysql-client default-libmysqlclient-dev gettext \
    libmagic1 ffmpeg libsm6 libxext6 libgl1-mesa-glx cron supervisor -yq --no-install-recommends

# copy source and install dependencies
RUN mkdir -p /usr/app/re_water
WORKDIR /usr/app/re_water
COPY ./requirements.txt /usr/app/re_water/requirements.txt

# install python packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools --upgrade
RUN python3 -m pip install wheel
RUN python3 -m pip install -r requirements.txt


# Use base state
FROM base_python3.10
WORKDIR /opt/app/re_water
COPY . /opt/app/re_water

EXPOSE 80
CMD [ "gunicorn",  "main:app", "--workers=4", "--worker-class=uvicorn.workers.UvicornWorker", "--bind=0.0.0.0:8000" ]
