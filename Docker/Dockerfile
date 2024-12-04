FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN apt-get install -y git && \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    DEBIAN_FRONTEND=noninteractive apt-get install gconf-service libasound2 \
    libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 \
    libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 \
    libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 \
    libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 \
    libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 \
    libnss3 lsb-release xdg-utils wget -y


WORKDIR /save
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pip && pip install -U openmim && mim install mmcv-full==1.4.0 && \
    pip install mmdet==2.24.1

RUN git clone https://github.com/open-mmlab/mmdetection3d.git && \
    cd mmdetection3d && git checkout v0.17.1 && pip install -r requirements/build.txt && \
    python3 setup.py develop





