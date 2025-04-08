# Use Miniconda as the base image to manage Python and dependencies
FROM continuumio/miniconda3:4.10.3 as base

# ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system packages required for v4l2-ctl, OpenGL support, and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    v4l-utils \
    libzbar0 \
    libgl1-mesa-glx \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libqt5widgets5 \
    libqt5gui5 \
    libqt5core5a \
    libqt5dbus5 \
    qt5-gtk-platformtheme \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ENV QT_QPA_PLATFORM='offscreen'
# Copy the environment.yml file into the container and create the Conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the Conda environment
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Ensure commands run inside the Conda environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copy the application's code into the container
COPY new_model_QR.pt .
COPY LED_on_off_model.pt .
COPY . .

# Expose the port the app runs on
EXPOSE 8070

# Specify the command to run the application
ENTRYPOINT ["conda", "run", "-n", "myenv", "python", "train_all_together.py"]
CMD ["automate_detection"]
