# disinfo-research-group-template


## Installation

### Repo Setup:

Make a copy of this template repo. Clone / download your copy of the repo.

And navigate there from the command-line.

```sh
cd ~/path/to/disinfo-research-group-template
```

### Environment Setup

> If you use conda environments, follow this section


Setup a virtual environment:

```sh
conda create -n research-env python=3.8
```

```sh
conda activate research-env
```

### Package Installation

Install packages:

```sh
pip install -r requirements.txt
```


## Setup

### Google Credentials

Gain access to the shared drive.

Locate the JSON credentials file in the "credentials" directory, and download it to the root directory of this repo, naming it specifically "google-credentials.json".

### Environment Variables

Create a new local ".env" file and set the following environment variable (`GOOGLE_APPLICATION_CREDENTIALS`):

```sh
# this is the ".env" file..

GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/disinfo-research-group-template/google-credentials.json"
```

## Usage


Demonstrate ability to fetch data from BigQuery:

```sh
python -m app.bq_service
```
