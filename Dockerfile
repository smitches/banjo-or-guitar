FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

# Install pytorch and fastai
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install fastai

# Install starlette and uvicorn
RUN pip install starlette uvicorn python-multipart aiohttp

ADD guitar.py guitar.py
RUN wget https://smitches-banjo.s3.us-east-2.amazonaws.com/banjo-or-guitar.pth
# ADD banjo-or-guitar.pth banjo-or-guitar.pth

# Run it once to trigger resnet download
RUN python guitar.py


EXPOSE 8008

# Start the server
CMD ["python", "guitar.py", "serve"]
