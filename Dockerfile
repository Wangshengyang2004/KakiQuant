# Use an official Python runtime as a parent image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Python3.10 and pip
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip

# Install build-essential for building Python packages
RUN apt-get install -y build-essential
# Install any needed packages specified in requirements.txt and requirements-gpu.txt
# Note: Uncomment the next line if GPU support is required and you have the appropriate base image
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -e .

# Run the script to install any additional requirements
# RUN chmod +x ./install-requirements.sh && \
#     ./install-requirements.sh

# Install CPU-based requirements (comment this out if using GPU requirements exclusively)
# RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80
EXPOSE 443
EXPOSE 8000

# Define environment variable
# Note: Replace values with your actual environment variables
# ENV NAME Value

# Run main.py when the container launches
CMD ["python3", "./kaki/main.py"]
