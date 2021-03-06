# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src/ .

# Set flask app variable
ENV FLASK_APP server

# command to run on container start
CMD [ "python", "-m", "flask", "run", "--host=0.0.0.0", "--cert=cert.pem", "--key=privkey.pem"]
